"""CSV generation routes - Spacy and Vision LLM extraction."""
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, ProcessingJob, ProjectStatus
from app.config import UPLOAD_DIR, DATABASE_URL

router = APIRouter(prefix="/api/generate", tags=["Generate"])


class GenerateRequest(BaseModel):
    method: str  # 'spacy' or 'vision'
    files: List[str]


def process_spacy_task(
    job_id: int,
    file_paths: List[str],
    project_id: int,
    settings: dict,
    db_url: str
):
    """Background task for Spacy text extraction."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.services.document_processor import process_documents_to_csv
    
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
        
        valid_count, csv_path = process_documents_to_csv(
            file_paths,
            project.folder_path,
            project.dataset_type.value,
            progress_callback,
            min_len=settings.get('min_sentence_length', 30),
            max_len=settings.get('max_sentence_length', 100),
            min_words=settings.get('min_words', 5)
        )
        
        project.total_entries = valid_count
        
        job.status = ProjectStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.message = f"Extracted {valid_count} sentences"
        db.commit()
        
    except Exception as e:
        job.status = ProjectStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def process_vision_task(
    job_id: int,
    file_paths: List[str],
    project_id: int,
    settings: dict,
    db_url: str
):
    """Background task for Vision LLM extraction."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.services.vision_processor import process_pdf_with_vision
    
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
        
        total_valid = 0
        
        for i, pdf_path in enumerate(file_paths):
            def progress_callback(progress: int, message: str):
                overall = int((i / len(file_paths)) * 100 + (progress / len(file_paths)))
                job.progress = overall
                job.message = f"File {i+1}/{len(file_paths)}: {message}"
                db.commit()
            
            valid_count, csv_path = process_pdf_with_vision(
                pdf_path,
                project.folder_path,
                project.dataset_type.value,
                progress_callback,
                provider=settings.get('ai_provider', 'ollama'),
                model=settings.get('ollama_model') if settings.get('ai_provider') == 'ollama' else settings.get('openai_model'),
                api_key=settings.get('openai_api_key')
            )
            total_valid += valid_count
        
        project.total_entries = total_valid
        
        job.status = ProjectStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.message = f"Extracted {total_valid} segments with Vision AI"
        db.commit()
        
    except Exception as e:
        job.status = ProjectStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


@router.post("/{project_id}")
async def generate_csv(
    project_id: int,
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start CSV generation from uploaded files."""
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
    
    # Get settings
    from app.models import ProjectSettings
    settings_obj = db.query(ProjectSettings).filter(
        ProjectSettings.project_id == project_id
    ).first()
    settings = settings_obj.to_dict() if settings_obj else {}
    
    # Create job
    job_type = 'vision' if request.method == 'vision' else 'spacy'
    job = ProcessingJob(
        job_type=job_type,
        project_id=project_id,
        user_id=current_user.id,
        status=ProjectStatus.PENDING,
        message="Queued for processing"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start background task
    if request.method == 'vision':
        # Filter only PDFs for vision
        pdf_paths = [p for p in file_paths if p.lower().endswith('.pdf')]
        if not pdf_paths:
            raise HTTPException(status_code=400, detail="Vision mode requires PDF files")
        background_tasks.add_task(
            process_vision_task, job.id, pdf_paths, project_id, settings, DATABASE_URL
        )
    else:
        background_tasks.add_task(
            process_spacy_task, job.id, file_paths, project_id, settings, DATABASE_URL
        )
    
    return {
        "message": "Generation started",
        "job_id": job.id
    }

