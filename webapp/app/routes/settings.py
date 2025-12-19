"""Project settings routes."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, ProjectSettings

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SettingsUpdate(BaseModel):
    # Text extraction
    min_sentence_length: Optional[int] = 30
    max_sentence_length: Optional[int] = 100
    min_words: Optional[int] = 5
    
    # Vision LLM
    ai_provider: Optional[str] = "ollama"
    ollama_model: Optional[str] = "gemma3:4b"
    openai_model: Optional[str] = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # Whisper
    whisper_model: Optional[str] = "iRaduS/whisper-romanian-finetune"
    min_segment_duration: Optional[int] = 3
    max_segment_duration: Optional[int] = 10
    
    # Recording
    sample_rate: Optional[int] = 44100
    silence_threshold: Optional[int] = 30
    auto_trim: Optional[bool] = True


@router.get("/{project_id}")
async def get_settings(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project settings."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    settings = db.query(ProjectSettings).filter(
        ProjectSettings.project_id == project_id
    ).first()
    
    if not settings:
        # Return defaults
        return SettingsUpdate().dict()
    
    return settings.to_dict()


@router.post("/{project_id}")
async def save_settings(
    project_id: int,
    settings_data: SettingsUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save project settings."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    settings = db.query(ProjectSettings).filter(
        ProjectSettings.project_id == project_id
    ).first()
    
    if not settings:
        settings = ProjectSettings(project_id=project_id)
        db.add(settings)
    
    # Update settings
    settings.min_sentence_length = settings_data.min_sentence_length
    settings.max_sentence_length = settings_data.max_sentence_length
    settings.min_words = settings_data.min_words
    settings.ai_provider = settings_data.ai_provider
    settings.ollama_model = settings_data.ollama_model
    settings.openai_model = settings_data.openai_model
    settings.openai_api_key = settings_data.openai_api_key
    settings.whisper_model = settings_data.whisper_model
    settings.min_segment_duration = settings_data.min_segment_duration
    settings.max_segment_duration = settings_data.max_segment_duration
    settings.sample_rate = settings_data.sample_rate
    settings.silence_threshold = settings_data.silence_threshold
    settings.auto_trim = settings_data.auto_trim
    
    db.commit()
    
    return {"message": "Settings saved"}

