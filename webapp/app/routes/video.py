"""Video to dataset routes."""
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, ProcessingJob, ProjectStatus
from app.config import UPLOAD_DIR, DATABASE_URL

router = APIRouter(prefix="/api/video", tags=["Video"])


class VideoProcessRequest(BaseModel):
    files: List[str]
    whisper_model: str = "iRaduS/whisper-romanian-finetune"
    min_duration: int = 3
    max_duration: int = 10
    min_silence_duration: float = 0.5  # Minimum pause to consider split
    padding_duration: float = 0.2  # Silence added at start/end
    silence_threshold: int = 45  # dB threshold for silence detection


def process_videos_task(
    job_id: int,
    file_paths: List[str],
    project_id: int,
    whisper_model: str,
    min_duration: int,
    max_duration: int,
    min_silence_duration: float,
    padding_duration: float,
    silence_threshold: int,
    db_url: str
):
    """Background task for video processing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.services.video_processor import process_videos_to_dataset
    
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        project = db.query(Project).filter(Project.id == project_id).first()
        
        if not job or not project:
            return
        
        job.status = ProjectStatus.PROCESSING
        job.started_at = datetime.utcnow()
        db.commit()
        
        def progress_callback(progress: int, message: str):
            job.progress = progress
            job.message = message
            db.commit()
        
        entries_added, csv_path = process_videos_to_dataset(
            file_paths,
            project.folder_path,
            project.dataset_type.value,
            progress_callback,
            whisper_model=whisper_model,
            min_dur=min_duration,
            max_dur=max_duration,
            min_silence_duration=min_silence_duration,
            padding_duration=padding_duration,
            silence_threshold=silence_threshold
        )
        
        # Load entries from CSV into database (append, don't delete)
        job.message = "Saving entries to database..."
        db.commit()
        
        import csv as csv_module
        import os
        from app.models import DatasetEntry
        
        csv_path = os.path.join(project.folder_path, 'metadata.csv')
        
        if os.path.exists(csv_path):
            # Get existing wav_filenames to avoid duplicates
            existing_filenames = set(
                entry.wav_filename for entry in 
                db.query(DatasetEntry.wav_filename).filter(DatasetEntry.project_id == project_id).all()
            )
            
            # Read CSV and insert only NEW entries
            db_entries_added = 0
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv_module.reader(f, delimiter='|')
                for row in reader:
                    if len(row) >= 2:
                        wav_filename = row[0]
                        # Skip if already in database
                        if wav_filename in existing_filenames:
                            continue
                        
                        # Check if audio file exists
                        audio_file_path = os.path.join(project.folder_path, wav_filename)
                        audio_exists = os.path.exists(audio_file_path)
                        
                        entry = DatasetEntry(
                            project_id=project_id,
                            wav_filename=wav_filename,
                            original_text=row[1],
                            normalized_text=row[2] if len(row) > 2 else row[1],
                            has_audio=audio_exists
                        )
                        db.add(entry)
                        db_entries_added += 1
            
            db.commit()
        
        # Update project counts
        total_entries = db.query(DatasetEntry).filter(DatasetEntry.project_id == project_id).count()
        recorded_entries = db.query(DatasetEntry).filter(
            DatasetEntry.project_id == project_id,
            DatasetEntry.has_audio == True
        ).count()
        
        project.total_entries = total_entries
        project.recorded_entries = recorded_entries
        
        job.status = ProjectStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.message = f"Added {entries_added} segments (total: {total_entries})"
        db.commit()
        
    except Exception as e:
        job.status = ProjectStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


@router.post("/{project_id}/process")
async def process_videos(
    project_id: int,
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start video processing with Whisper."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not request.files:
        raise HTTPException(status_code=400, detail="No files selected")
    
    # Build full paths
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    file_paths = []
    for filename in request.files:
        file_path = upload_folder / filename
        if file_path.exists():
            file_paths.append(str(file_path))
    
    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid files found")
    
    # Create job
    job = ProcessingJob(
        job_type='video',
        project_id=project_id,
        user_id=current_user.id,
        status=ProjectStatus.PENDING,
        message="Queued for video processing"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start background task
    background_tasks.add_task(
        process_videos_task,
        job.id,
        file_paths,
        project_id,
        request.whisper_model,
        request.min_duration,
        request.max_duration,
        request.min_silence_duration,
        request.padding_duration,
        request.silence_threshold,
        DATABASE_URL
    )
    
    return {
        "message": "Video processing started",
        "job_id": job.id
    }

