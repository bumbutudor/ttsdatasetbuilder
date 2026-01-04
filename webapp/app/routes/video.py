"""Video processing routes with multi-step workflow."""
import os
import shutil
import json
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, ProcessingJob, ProjectStatus, DatasetEntry, DatasetType
from app.config import UPLOAD_DIR, PROJECTS_DIR, DATABASE_URL
from app.services.video_processor import check_model_exists, download_model_task, split_audio_staging, transcribe_staging

router = APIRouter(prefix="/api/video", tags=["Video"])

# Logger
logger = logging.getLogger(__name__)

# --- Models ---
class ModelCheckRequest(BaseModel):
    model_name: str

class SplitRequest(BaseModel):
    files: List[str]

class TranscribeRequest(BaseModel):
    staging_id: str
    files: List[str] # Filenames in staging
    whisper_model: str

class CommitRequest(BaseModel):
    staging_id: str
    entries: List[dict] # {filename, text}

# --- State Management Helpers ---
# In a production app, use Redis or DB. For this local app, file system + Job DB is fine.
def get_staging_dir(project_id: int, staging_id: str):
    return PROJECTS_DIR / f"staging_{project_id}_{staging_id}"

# --- Endpoints ---

@router.post("/check-model")
async def check_model(request: ModelCheckRequest, current_user: User = Depends(get_current_active_user)):
    exists = check_model_exists(request.model_name)
    return {"exists": exists}

@router.post("/download-model")
async def download_model(
    request: ModelCheckRequest, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Create a system job for download
    job = ProcessingJob(
        job_type='model_download',
        project_id=0, # System job
        user_id=current_user.id,
        status=ProjectStatus.PENDING,
        message=f"Downloading {request.model_name}..."
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    def task_wrapper(job_id):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        job = db.query(ProcessingJob).get(job_id)
        
        def progress(p, msg):
            job.progress = p
            job.message = msg
            db.commit()
            
        try:
            job.status = ProjectStatus.PROCESSING
            download_model_task(request.model_name, progress)
            job.status = ProjectStatus.COMPLETED
        except Exception as e:
            job.status = ProjectStatus.FAILED
            job.error_message = str(e)
        finally:
            db.commit()
            db.close()

    background_tasks.add_task(task_wrapper, job.id)
    return {"job_id": job.id}

@router.post("/{project_id}/split")
async def split_audio(
    project_id: int,
    request: SplitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Decide segmentation based on PROJECT dataset type
    dataset_type = 'stt' if project.dataset_type == DatasetType.STT else 'tts'

    # --- LOGGING DEBUG ---
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    logger.info(f"DEBUG: Looking for files in: {upload_folder}")
    logger.info(f"DEBUG: Requested files: {request.files}")
    
    video_paths = []
    for f in request.files:
        full_path = upload_folder / f
        if full_path.exists():
            video_paths.append(str(full_path))
        else:
            logger.error(f"DEBUG: File NOT FOUND: {full_path}")
            
    if not video_paths:
        raise HTTPException(status_code=400, detail="No valid video files found on server.")
        
    logger.info(f"DEBUG: Valid video paths to process: {video_paths}")
    # ---------------------

    # Create Job
    job = ProcessingJob(
        job_type='video_split',
        project_id=project_id,
        user_id=current_user.id,
        status=ProjectStatus.PENDING,
        message="Initializing split..."
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Staging ID is the Job ID
    staging_dir = get_staging_dir(project_id, str(job.id))
    
    # Get file paths
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    video_paths = [str(upload_folder / f) for f in request.files if (upload_folder / f).exists()]

    def task_wrapper(job_id, v_paths, s_dir, ds_type: str):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        # Re-import logger for thread context
        import logging as _logging
        logger = _logging.getLogger(__name__)

        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        local_db = SessionLocal()
        job_ref = local_db.query(ProcessingJob).get(job_id)
        
        def progress(p, msg):
            logger.info(f"JOB PROGRESS {p}%: {msg}")
            job_ref.progress = p
            job_ref.message = msg
            local_db.commit()
            
        def check_cancel():
            local_db.refresh(job_ref)
            return job_ref.status == ProjectStatus.FAILED
            
        try:
            job_ref.status = ProjectStatus.PROCESSING
            local_db.commit()

            logger.info(f"STARTING SPLIT LOGIC with dataset_type={ds_type}")

            segments = split_audio_staging(
                v_paths,
                str(s_dir),
                ds_type,
                progress,
                check_cancel
            )
            
            logger.info(f"SPLIT COMPLETE. Found {len(segments)} segments.")

            # Save segments list to a JSON file
            with open(s_dir / "segments.json", "w") as f:
                json.dump(segments, f)
                
            job_ref.status = ProjectStatus.COMPLETED
            job_ref.message = f"Split complete. Found {len(segments)} segments."
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"JOB FAILED: {str(e)}\n{error_trace}")
            
            job_ref.status = ProjectStatus.FAILED
            job_ref.error_message = f"Error: {str(e)}"
        finally:
            local_db.commit()
            local_db.close()

    background_tasks.add_task(task_wrapper, job.id, video_paths, staging_dir, dataset_type)
    return {"job_id": job.id, "staging_id": str(job.id)}

@router.get("/{project_id}/staging/{staging_id}/segments")
async def get_segments(project_id: int, staging_id: str, current_user: User = Depends(get_current_active_user)):
    staging_dir = get_staging_dir(project_id, staging_id)
    json_path = staging_dir / "segments.json"
    
    if not json_path.exists():
        return {"segments": []}
        
    with open(json_path, "r") as f:
        return {"segments": json.load(f)}

@router.get("/{project_id}/staging/{staging_id}/audio/{filename}")
async def get_staging_audio(project_id: int, staging_id: str, filename: str):
    path = get_staging_dir(project_id, staging_id) / filename
    if path.exists():
        return FileResponse(str(path))
    raise HTTPException(404)

@router.delete("/{project_id}/staging/{staging_id}/segment/{filename}")
async def delete_segment(project_id: int, staging_id: str, filename: str):
    path = get_staging_dir(project_id, staging_id) / filename
    if path.exists():
        os.remove(path)
        # Update JSON
        json_path = get_staging_dir(project_id, staging_id) / "segments.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                segs = json.load(f)
            segs = [s for s in segs if s['filename'] != filename]
            with open(json_path, "w") as f:
                json.dump(segs, f)
    return {"status": "deleted"}

@router.post("/{project_id}/transcribe")
async def start_transcribe(
    project_id: int,
    request: TranscribeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    job = ProcessingJob(
        job_type='video_transcribe',
        project_id=project_id,
        user_id=current_user.id,
        status=ProjectStatus.PENDING,
        message="Initializing transcription..."
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    staging_dir = get_staging_dir(project_id, request.staging_id)

    def task_wrapper(job_id, s_dir, files, model):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        local_db = SessionLocal()
        job_ref = local_db.query(ProcessingJob).get(job_id)
        
        def progress(p, msg):
            job_ref.progress = p
            job_ref.message = msg
            local_db.commit()
        
        def check_cancel():
            local_db.refresh(job_ref)
            return job_ref.status == ProjectStatus.FAILED

        try:
            job_ref.status = ProjectStatus.PROCESSING
            local_db.commit()
            
            results = transcribe_staging(
                str(s_dir), files, model, progress, check_cancel
            )
            
            # Save results to JSON
            with open(s_dir / "transcriptions.json", "w") as f:
                json.dump(results, f)
                
            job_ref.status = ProjectStatus.COMPLETED
            job_ref.message = "Transcription complete. Review text."
        except Exception as e:
            job_ref.status = ProjectStatus.FAILED
            job_ref.error_message = str(e)
        finally:
            local_db.commit()
            local_db.close()

    background_tasks.add_task(task_wrapper, job.id, staging_dir, request.files, request.whisper_model)
    return {"job_id": job.id}

@router.get("/{project_id}/staging/{staging_id}/transcriptions")
async def get_transcriptions(project_id: int, staging_id: str):
    path = get_staging_dir(project_id, staging_id) / "transcriptions.json"
    if path.exists():
        with open(path, "r") as f:
            return {"results": json.load(f)}
    return {"results": []}

@router.post("/{project_id}/cancel/{job_id}")
async def cancel_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ProcessingJob).get(job_id)
    if job and job.status in [ProjectStatus.PENDING, ProjectStatus.PROCESSING]:
        job.status = ProjectStatus.FAILED
        job.error_message = "Cancelled by user"
        db.commit()
    return {"status": "cancelled"}

@router.post("/{project_id}/commit")
async def commit_dataset(
    project_id: int, 
    request: CommitRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).get(project_id)
    staging_dir = get_staging_dir(project_id, request.staging_id)
    
    if not project.folder_path:
        # Init folder logic if missing
        pass # (Assumed existing)

    dest_folder = Path(project.folder_path)
    os.makedirs(dest_folder, exist_ok=True)
    
    # Determine start index
    next_index = 0
    import glob
    existing_wavs = glob.glob(str(dest_folder / "*.wav"))
    for w in existing_wavs:
        try:
            idx = int(Path(w).stem)
            next_index = max(next_index, idx + 1)
        except: pass
        
    entries_added = 0
    csv_path = dest_folder / "metadata.csv"
    
    with open(csv_path, "a", encoding="utf-8", newline='') as f:
        import csv
        writer = csv.writer(f, delimiter='|')
        
        for item in request.entries:
            src_path = staging_dir / item['filename']
            if not src_path.exists(): continue
            
            new_filename = f"{next_index:012d}.wav"
            dst_path = dest_folder / new_filename
            
            shutil.copy2(src_path, dst_path)
            
            writer.writerow([new_filename, item['text'], item['text']]) # Raw | Norm
            
            # DB Entry
            entry = DatasetEntry(
                project_id=project_id,
                wav_filename=new_filename,
                original_text=item['text'],
                normalized_text=item['text'],
                has_audio=True
            )
            db.add(entry)
            
            next_index += 1
            entries_added += 1
            
    project.total_entries += entries_added
    project.recorded_entries += entries_added
    db.commit()
    
    # Cleanup staging
    shutil.rmtree(staging_dir, ignore_errors=True)
    
    return {"added": entries_added}