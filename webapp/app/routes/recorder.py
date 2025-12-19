"""Voice recorder routes."""
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, DatasetEntry

router = APIRouter(prefix="/api/recorder", tags=["Recorder"])


@router.get("/{project_id}/sentences")
async def get_sentences(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all sentences for recording."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    entries = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id
    ).order_by(DatasetEntry.id).all()
    
    # Check which entries have audio files
    project_folder = Path(project.folder_path) if project.folder_path else None
    
    sentences = []
    recorded_count = 0
    
    for entry in entries:
        has_audio = False
        if project_folder:
            audio_path = project_folder / entry.wav_filename
            has_audio = audio_path.exists()
        
        if has_audio:
            recorded_count += 1
        
        sentences.append({
            "id": entry.id,
            "wav_filename": entry.wav_filename,
            "text": entry.normalized_text or entry.original_text,
            "has_audio": has_audio
        })
    
    return {
        "sentences": sentences,
        "total": len(entries),
        "recorded": recorded_count,
        "pending": len(entries) - recorded_count
    }


@router.post("/{project_id}/upload/{entry_id}")
async def upload_recording(
    project_id: int,
    entry_id: int,
    audio: UploadFile = File(...),
    auto_trim: bool = Form(True),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload a voice recording for a sentence."""
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
    
    # Save audio file
    project_folder = Path(project.folder_path)
    project_folder.mkdir(parents=True, exist_ok=True)
    
    audio_path = project_folder / entry.wav_filename
    
    # Save uploaded file
    with open(audio_path, "wb") as f:
        content = await audio.read()
        f.write(content)
    
    # Optional: trim silence
    if auto_trim:
        try:
            from app.services.audio_processor import trim_silence
            trim_silence(str(audio_path), project.dataset_type.value)
        except Exception as e:
            # Don't fail if trimming fails
            pass
    
    # Update entry
    entry.has_audio = True
    db.commit()
    
    # Update project stats
    recorded_count = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id,
        DatasetEntry.has_audio == True
    ).count()
    project.recorded_entries = recorded_count
    db.commit()
    
    return {
        "message": "Recording uploaded",
        "filename": entry.wav_filename
    }


@router.get("/{project_id}/audio/{filename}")
async def get_audio(
    project_id: int,
    filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get an audio file."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    audio_path = Path(project.folder_path) / filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(
        path=str(audio_path),
        filename=filename,
        media_type="audio/wav"
    )

