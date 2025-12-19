"""Text normalization routes."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, DatasetEntry

router = APIRouter(prefix="/api/normalize", tags=["Normalize"])


class NormalizeRequest(BaseModel):
    type: str  # 'tts' or 'stt'


@router.post("/{project_id}")
async def normalize_dataset(
    project_id: int,
    request: NormalizeRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Apply normalization to dataset entries."""
    from app.services.text_normalizer import normalize_for_tts, normalize_for_stt
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get all entries
    entries = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id
    ).all()
    
    if not entries:
        raise HTTPException(status_code=400, detail="No entries to normalize")
    
    # Apply normalization
    normalize_func = normalize_for_tts if request.type == 'tts' else normalize_for_stt
    
    normalized_count = 0
    for entry in entries:
        try:
            entry.normalized_text = normalize_func(entry.original_text)
            normalized_count += 1
        except Exception as e:
            # Skip entries that fail normalization
            continue
    
    db.commit()
    
    return {
        "message": f"Normalized {normalized_count} entries",
        "normalized_count": normalized_count,
        "type": request.type
    }

