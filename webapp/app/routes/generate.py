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
    project_folder: str,
    settings: dict,
    db_url: str
):
    """Background task for Spacy text extraction."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import os
    
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
        job.message = "Starting text extraction..."
        db.commit()
        
        # Ensure folder exists
        os.makedirs(project_folder, exist_ok=True)
        
        def progress_callback(progress: int, message: str):
            job.progress = progress
            job.message = message
            db.commit()
        
        # Import here to avoid circular imports
        from app.services.document_processor import process_documents_to_csv
        from app.models import DatasetEntry
        
        # Get the current highest index from database to continue from there
        existing_count = db.query(DatasetEntry).filter(DatasetEntry.project_id == project_id).count()
        
        valid_count, csv_path = process_documents_to_csv(
            file_paths,
            project_folder,
            project.dataset_type.value,
            progress_callback,
            min_len=settings.get('min_sentence_length', 30),
            max_len=settings.get('max_sentence_length', 100),
            min_words=settings.get('min_words', 5),
            start_index=existing_count
        )
        
        # Load entries from CSV into database (append, don't delete existing)
        job.message = "Saving entries to database..."
        db.commit()
        
        import csv as csv_module
        from app.models import DatasetEntry
        
        # Get existing wav_filenames to avoid duplicates
        existing_filenames = set(
            entry.wav_filename for entry in 
            db.query(DatasetEntry.wav_filename).filter(DatasetEntry.project_id == project_id).all()
        )
        
        # Read CSV and insert only NEW entries
        entries_added = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_module.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 2:
                    wav_filename = row[0]
                    # Skip if already in database
                    if wav_filename in existing_filenames:
                        continue
                    
                    # Check if audio file exists
                    audio_file_path = os.path.join(project_folder, wav_filename)
                    audio_exists = os.path.exists(audio_file_path)
                    
                    entry = DatasetEntry(
                        project_id=project_id,
                        wav_filename=wav_filename,
                        original_text=row[1],
                        normalized_text=row[2] if len(row) > 2 else row[1],
                        has_audio=audio_exists
                    )
                    db.add(entry)
                    entries_added += 1
        
        db.commit()
        
        # Update project total count
        total_entries = db.query(DatasetEntry).filter(DatasetEntry.project_id == project_id).count()
        project.total_entries = total_entries
        
        job.status = ProjectStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.message = f"Added {entries_added} new sentences (total: {total_entries})"
        db.commit()
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        job.status = ProjectStatus.FAILED
        job.error_message = error_detail[:1000]  # Limit error message length
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def process_vision_task(
    job_id: int,
    file_paths: List[str],
    project_id: int,
    project_folder: str,
    settings: dict,
    db_url: str
):
    """Background task for Vision LLM extraction."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Vision task job_id={job_id}, project_id={project_id}")
    
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        project = db.query(Project).filter(Project.id == project_id).first()
        
        if not job or not project:
            logger.error(f"Job or project not found: job={job}, project={project}")
            return
        
        logger.info(f"Processing {len(file_paths)} files for project folder: {project_folder}")
        
        job.status = ProjectStatus.PROCESSING
        job.started_at = datetime.utcnow()
        job.message = "Starting Vision LLM extraction..."
        db.commit()
        
        # Ensure folder exists
        os.makedirs(project_folder, exist_ok=True)
        
        # Import here to avoid circular imports
        from app.services.vision_processor import process_pdf_with_vision
        
        total_valid = 0
        
        for i, pdf_path in enumerate(file_paths):
            def progress_callback(progress: int, message: str):
                overall = int((i / len(file_paths)) * 100 + (progress / len(file_paths)))
                job.progress = overall
                job.message = f"File {i+1}/{len(file_paths)}: {message}"
                db.commit()
            
            valid_count, csv_path = process_pdf_with_vision(
                pdf_path,
                project_folder,
                project.dataset_type.value,
                progress_callback,
                provider=settings.get('ai_provider', 'ollama'),
                model=settings.get('ollama_model') if settings.get('ai_provider') == 'ollama' else settings.get('openai_model'),
                api_key=settings.get('openai_api_key')
            )
            total_valid += valid_count
        
        # Load entries from CSV into database (append, don't delete existing)
        job.message = "Saving entries to database..."
        db.commit()
        
        import csv as csv_module
        from app.models import DatasetEntry
        
        csv_path = os.path.join(project_folder, 'metadata.csv')
        logger.info(f"Looking for CSV at: {csv_path}")
        
        entries_added = 0
        
        if os.path.exists(csv_path):
            logger.info(f"CSV found, reading entries...")
            
            # Get existing wav_filenames to avoid duplicates
            existing_entries = db.query(DatasetEntry.wav_filename).filter(
                DatasetEntry.project_id == project_id
            ).all()
            existing_filenames = set(e.wav_filename for e in existing_entries)
            logger.info(f"Found {len(existing_filenames)} existing entries in DB")
            
            # Read CSV and insert only NEW entries
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv_module.reader(f, delimiter='|')
                rows = list(reader)
                logger.info(f"CSV has {len(rows)} rows")
                
                for row in rows:
                    if len(row) >= 2:
                        wav_filename = row[0]
                        # Skip if already in database
                        if wav_filename in existing_filenames:
                            continue
                        
                        # Check if audio file exists
                        audio_file_path = os.path.join(project_folder, wav_filename)
                        audio_exists = os.path.exists(audio_file_path)
                        
                        entry = DatasetEntry(
                            project_id=project_id,
                            wav_filename=wav_filename,
                            original_text=row[1],
                            normalized_text=row[2] if len(row) > 2 else row[1],
                            has_audio=audio_exists
                        )
                        db.add(entry)
                        entries_added += 1
            
            logger.info(f"Added {entries_added} new entries to DB")
            db.commit()
            logger.info("DB committed successfully")
        else:
            logger.error(f"CSV not found at {csv_path}")
        
        # Update project total count
        total_entries = db.query(DatasetEntry).filter(DatasetEntry.project_id == project_id).count()
        project.total_entries = total_entries
        logger.info(f"Total entries in project: {total_entries}")
        
        job.status = ProjectStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.message = f"Added {entries_added} new segments with Vision AI (total: {total_entries})"
        db.commit()
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        job.status = ProjectStatus.FAILED
        job.error_message = error_detail[:1000]
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
    from app.models import ProjectSettings
    from app.config import PROJECTS_DIR
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not request.files:
        raise HTTPException(status_code=400, detail="No files selected")
    
    # Ensure project folder exists
    if not project.folder_path:
        from datetime import datetime
        folder_name = f"project_{project.dataset_type.value}_{project.id}_{datetime.now().strftime('%Y%m%d')}"
        folder_path = PROJECTS_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        project.folder_path = str(folder_path)
        db.commit()
    
    # Build full paths
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    file_paths = []
    for filename in request.files:
        file_path = upload_folder / filename
        if file_path.exists():
            file_paths.append(str(file_path))
    
    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid files found")
    
    # Get settings or use defaults
    settings_obj = db.query(ProjectSettings).filter(
        ProjectSettings.project_id == project_id
    ).first()
    settings = settings_obj.to_dict() if settings_obj else {
        'min_sentence_length': 30,
        'max_sentence_length': 100,
        'min_words': 5,
        'ai_provider': 'ollama',
        'ollama_model': 'gemma3:4b',
        'openai_model': 'gpt-4o-mini'
    }
    
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
            process_vision_task, job.id, pdf_paths, project_id, project.folder_path, settings, DATABASE_URL
        )
    else:
        background_tasks.add_task(
            process_spacy_task, job.id, file_paths, project_id, project.folder_path, settings, DATABASE_URL
        )
    
    return {
        "message": "Generation started",
        "job_id": job.id
    }

