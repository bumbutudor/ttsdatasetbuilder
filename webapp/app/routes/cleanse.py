"""Dataset cleansing routes."""
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, DatasetEntry

router = APIRouter(prefix="/api/cleanse", tags=["Cleanse"])


@router.get("/{project_id}/stats")
async def get_cleanse_stats(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get dataset statistics for cleansing."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    entries = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id
    ).all()
    
    project_folder = Path(project.folder_path) if project.folder_path else None
    
    with_audio = 0
    missing_audio = 0
    total_size = 0
    
    for entry in entries:
        if project_folder:
            audio_path = project_folder / entry.wav_filename
            if audio_path.exists():
                with_audio += 1
                total_size += audio_path.stat().st_size
            else:
                missing_audio += 1
        else:
            missing_audio += 1
    
    return {
        "total_entries": len(entries),
        "with_audio": with_audio,
        "missing_audio": missing_audio,
        "total_size_bytes": total_size
    }


@router.get("/{project_id}/missing")
async def get_missing_files(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get entries with missing audio files."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    entries = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id
    ).all()
    
    project_folder = Path(project.folder_path) if project.folder_path else None
    
    missing = []
    for entry in entries:
        has_audio = False
        if project_folder:
            audio_path = project_folder / entry.wav_filename
            has_audio = audio_path.exists()
        
        if not has_audio:
            missing.append({
                "id": entry.id,
                "wav_filename": entry.wav_filename,
                "original_text": entry.original_text
            })
    
    return {"missing": missing}


@router.post("/{project_id}")
async def cleanse_dataset(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Remove entries without audio files."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    entries = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id
    ).all()
    
    project_folder = Path(project.folder_path) if project.folder_path else None
    
    removed_count = 0
    valid_count = 0
    
    for entry in entries:
        has_audio = False
        if project_folder:
            audio_path = project_folder / entry.wav_filename
            has_audio = audio_path.exists()
        
        if has_audio:
            valid_count += 1
        else:
            db.delete(entry)
            removed_count += 1
    
    # Update project stats
    project.total_entries = valid_count
    project.recorded_entries = valid_count
    
    db.commit()
    
    return {
        "message": f"Removed {removed_count} entries, kept {valid_count}",
        "removed_count": removed_count,
        "valid_count": valid_count
    }


@router.delete("/{project_id}/entry/{entry_id}")
async def delete_entry(
    project_id: int,
    entry_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a single entry."""
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
    
    db.commit()
    
    return {"message": "Entry deleted"}

