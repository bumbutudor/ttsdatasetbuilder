"""Project management routes."""
import os
import csv
import shutil
import zipfile
import tempfile
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, DatasetEntry, DatasetType, ProjectStatus
from app.schemas import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    DatasetEntryListResponse,
    DatasetEntryResponse,
    ProjectStats,
)
from app.config import PROJECTS_DIR, UPLOAD_DIR

router = APIRouter(prefix="/api/projects", tags=["Projects"])


def get_project_folder(project: Project) -> Path:
    """Get or create project folder path."""
    if project.folder_path and os.path.exists(project.folder_path):
        return Path(project.folder_path)
    
    folder_name = f"project_{project.dataset_type.value}_{project.id}_{datetime.now().strftime('%Y%m%d')}"
    folder_path = PROJECTS_DIR / folder_name
    folder_path.mkdir(exist_ok=True)
    
    return folder_path


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all projects for the current user."""
    query = db.query(Project).filter(Project.owner_id == current_user.id)
    total = query.count()
    projects = query.offset(skip).limit(limit).all()
    
    return {"projects": projects, "total": total}


@router.post("", response_model=ProjectResponse)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new project."""
    project = Project(
        name=project_data.name,
        description=project_data.description,
        dataset_type=project_data.dataset_type,
        owner_id=current_user.id,
        status=ProjectStatus.PENDING
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    
    # Create project folder
    folder_path = get_project_folder(project)
    project.folder_path = str(folder_path)
    db.commit()
    
    return project


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project details."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return project


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update project details."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project_data.name is not None:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description
    
    db.commit()
    db.refresh(project)
    
    return project


@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a project and all its data."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Delete project folder
    if project.folder_path and os.path.exists(project.folder_path):
        shutil.rmtree(project.folder_path)
    
    # Delete upload folder
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    if upload_folder.exists():
        shutil.rmtree(upload_folder)
    
    db.delete(project)
    db.commit()
    
    return {"message": "Project deleted successfully"}


@router.get("/{project_id}/entries", response_model=DatasetEntryListResponse)
async def list_entries(
    project_id: int,
    page: int = 1,
    per_page: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List dataset entries for a project."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Try to load from database first
    query = db.query(DatasetEntry).filter(DatasetEntry.project_id == project_id)
    total = query.count()
    
    if total == 0 and project.folder_path:
        # Load from CSV if database is empty
        csv_path = os.path.join(project.folder_path, 'metadata.csv')
        if os.path.exists(csv_path):
            entries = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                for row in reader:
                    if len(row) >= 2:
                        wav_path = os.path.join(project.folder_path, row[0])
                        entry = DatasetEntry(
                            project_id=project_id,
                            wav_filename=row[0],
                            original_text=row[1],
                            normalized_text=row[2] if len(row) > 2 else row[1],
                            has_audio=os.path.exists(wav_path)
                        )
                        entries.append(entry)
            
            # Bulk insert
            if entries:
                db.add_all(entries)
                db.commit()
                
                # Update project stats
                project.total_entries = len(entries)
                project.recorded_entries = sum(1 for e in entries if e.has_audio)
                db.commit()
    
    # Get paginated entries
    offset = (page - 1) * per_page
    entries = query.offset(offset).limit(per_page).all()
    total = query.count()
    
    return {
        "entries": entries,
        "total": total,
        "page": page,
        "per_page": per_page
    }


@router.get("/{project_id}/stats", response_model=ProjectStats)
async def get_project_stats(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project statistics."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Count entries
    total = db.query(DatasetEntry).filter(DatasetEntry.project_id == project_id).count()
    recorded = db.query(DatasetEntry).filter(
        DatasetEntry.project_id == project_id,
        DatasetEntry.has_audio == True
    ).count()
    
    # Estimate durations (TTS ~4s, STT ~8s per entry)
    avg_duration = 4 if project.dataset_type == DatasetType.TTS else 8
    
    return ProjectStats(
        total_entries=total,
        recorded_entries=recorded,
        pending_entries=total - recorded,
        estimated_duration_hours=(total * avg_duration) / 3600,
        recorded_duration_hours=(recorded * avg_duration) / 3600
    )


@router.get("/{project_id}/download")
async def download_dataset(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Download project dataset as ZIP."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.folder_path or not os.path.exists(project.folder_path):
        raise HTTPException(status_code=404, detail="Project folder not found")
    
    # Create ZIP file
    zip_filename = f"{project.name.replace(' ', '_')}_{project.id}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project.folder_path)
                zipf.write(file_path, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=zip_filename
    )


@router.post("/{project_id}/cleanse")
async def cleanse_dataset(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Remove entries without corresponding audio files."""
    from app.services.document_processor import cleanse_csv
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.folder_path:
        raise HTTPException(status_code=400, detail="Project folder not set")
    
    try:
        valid_count, removed_count = cleanse_csv(project.folder_path)
        
        # Update database entries
        entries = db.query(DatasetEntry).filter(DatasetEntry.project_id == project_id).all()
        for entry in entries:
            wav_path = os.path.join(project.folder_path, entry.wav_filename)
            if not os.path.exists(wav_path):
                db.delete(entry)
        
        db.commit()
        
        # Update project stats
        project.total_entries = valid_count
        project.recorded_entries = valid_count
        db.commit()
        
        return {
            "message": f"Kept {valid_count} entries, removed {removed_count} missing files"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

