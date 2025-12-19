"""Dataset viewing and management routes."""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, DatasetEntry

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/{project_id}")
async def get_dataset(
    project_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    has_audio: Optional[str] = None,
    source: str = "original",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get dataset entries with pagination and filtering."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Base query
    query = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id
    )
    
    # Search filter
    if search:
        query = query.filter(
            or_(
                DatasetEntry.original_text.ilike(f"%{search}%"),
                DatasetEntry.normalized_text.ilike(f"%{search}%"),
                DatasetEntry.wav_filename.ilike(f"%{search}%")
            )
        )
    
    # Audio filter
    project_folder = Path(project.folder_path) if project.folder_path else None
    
    # Get total count
    total = query.count()
    
    # Paginate
    entries = query.order_by(DatasetEntry.id).offset(
        (page - 1) * per_page
    ).limit(per_page).all()
    
    # Build response
    result_entries = []
    for entry in entries:
        entry_has_audio = False
        if project_folder:
            audio_path = project_folder / entry.wav_filename
            entry_has_audio = audio_path.exists()
        
        # Filter by audio if requested
        if has_audio == "yes" and not entry_has_audio:
            continue
        if has_audio == "no" and entry_has_audio:
            continue
        
        result_entries.append({
            "id": entry.id,
            "wav_filename": entry.wav_filename,
            "original_text": entry.original_text,
            "normalized_text": entry.normalized_text,
            "has_audio": entry_has_audio
        })
    
    return {
        "entries": result_entries,
        "total": total,
        "page": page,
        "per_page": per_page
    }


@router.get("/{project_id}/audio/{filename}")
async def get_audio(
    project_id: int,
    filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get an audio file from the dataset."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not project.folder_path:
        raise HTTPException(status_code=404, detail="No project folder")
    
    audio_path = Path(project.folder_path) / filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(
        path=str(audio_path),
        filename=filename,
        media_type="audio/wav"
    )


@router.delete("/{project_id}/entry/{entry_id}")
async def delete_entry(
    project_id: int,
    entry_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a dataset entry and its audio file."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    entry = db.query(DatasetEntry).filter(
        DatasetEntry.id == entry_id,
        DatasetEntry.project_id == project_id
    ).first()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    # Delete audio file if exists
    if project.folder_path:
        audio_path = Path(project.folder_path) / entry.wav_filename
        if audio_path.exists():
            audio_path.unlink()
    
    db.delete(entry)
    
    # Update project stats
    project.total_entries = max(0, project.total_entries - 1)
    if entry.has_audio:
        project.recorded_entries = max(0, project.recorded_entries - 1)
    
    db.commit()
    
    return {"message": "Entry deleted"}

