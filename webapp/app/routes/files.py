"""File management routes - upload only, no processing."""
import os
import shutil
from typing import List
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project
from app.config import UPLOAD_DIR

router = APIRouter(prefix="/api/files", tags=["Files"])


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_media_type(filename: str) -> str:
    """Get MIME type for file."""
    ext = Path(filename).suffix.lower()
    types = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.mp4': 'video/mp4',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm'
    }
    return types.get(ext, 'application/octet-stream')


@router.get("/{project_id}")
async def list_files(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all uploaded files for a project."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    
    documents = []
    videos = []
    
    doc_extensions = {'.pdf', '.txt'}
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
    
    if upload_folder.exists():
        for file_path in upload_folder.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_info = {
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "size_formatted": format_file_size(file_path.stat().st_size),
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                if ext in doc_extensions:
                    file_info["type"] = "pdf" if ext == ".pdf" else "txt"
                    documents.append(file_info)
                elif ext in video_extensions:
                    file_info["type"] = ext[1:]
                    videos.append(file_info)
    
    return {
        "documents": sorted(documents, key=lambda x: x["name"]),
        "videos": sorted(videos, key=lambda x: x["name"])
    }


@router.post("/documents/{project_id}")
async def upload_documents(
    project_id: int,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload document files (PDF/TXT) - no processing."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file types
    allowed_extensions = {'.pdf', '.txt'}
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {ext}. Allowed: {allowed_extensions}"
            )
    
    # Save files
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for file in files:
        file_path = upload_folder / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_count += 1
    
    return {
        "message": f"{saved_count} documents uploaded",
        "files_uploaded": saved_count
    }


@router.post("/videos/{project_id}")
async def upload_videos(
    project_id: int,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload video files - no processing."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file types
    allowed_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {ext}. Allowed: {allowed_extensions}"
            )
    
    # Save files
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for file in files:
        file_path = upload_folder / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_count += 1
    
    return {
        "message": f"{saved_count} videos uploaded",
        "files_uploaded": saved_count
    }


@router.get("/{project_id}/view/{filename}")
async def view_file(
    project_id: int,
    filename: str,
    token: str = None,  # Accept token in query params for new tab opening
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """View/download an uploaded file."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = UPLOAD_DIR / f"project_{project_id}" / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=get_media_type(filename)
    )


@router.get("/{project_id}/download/{filename}")
async def download_file_with_token(
    project_id: int,
    filename: str,
    token: str,
    db: Session = Depends(get_db)
):
    """Download file using token in URL (for opening in new tab)."""
    from app.auth import get_user_from_token
    
    # Verify token
    user = get_user_from_token(token, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = UPLOAD_DIR / f"project_{project_id}" / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=get_media_type(filename)
    )


@router.delete("/{project_id}/{filename}")
async def delete_file(
    project_id: int,
    filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an uploaded file."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Security check
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = UPLOAD_DIR / f"project_{project_id}" / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return {"message": f"File '{filename}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

