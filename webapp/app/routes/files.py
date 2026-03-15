"""File management routes - upload only, no processing."""
import os
import shutil
from typing import List
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project
from app.config import UPLOAD_DIR

router = APIRouter(prefix="/api/files", tags=["Files"])

# Modifică setul de extensii permise (sus în fișier sau în funcții)
# Adăugăm extensii audio comune
MEDIA_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.flac', '.ogg', '.m4a'}

MAX_UPLOAD_FILENAME_STEM_LEN = 25


def _safe_uploaded_filename(upload_folder: Path, original_filename: str, max_stem_len: int = MAX_UPLOAD_FILENAME_STEM_LEN) -> str:
    """Return a safe filename with stem truncated and made unique inside upload_folder."""
    base_name = os.path.basename(original_filename or "")
    ext = Path(base_name).suffix
    stem = Path(base_name).stem

    # Keep only simple chars to avoid weird filesystem issues
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))
    safe_stem = (safe_stem or "file")[:max_stem_len]

    candidate = f"{safe_stem}{ext}"
    if not (upload_folder / candidate).exists():
        return candidate

    counter = 1
    while True:
        suffix = f"_{counter}"
        trimmed = safe_stem[: max(1, max_stem_len - len(suffix))]
        candidate = f"{trimmed}{suffix}{ext}"
        if not (upload_folder / candidate).exists():
            return candidate
        counter += 1

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
    video_extensions = MEDIA_EXTENSIONS
    
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
        safe_name = _safe_uploaded_filename(upload_folder, file.filename)
        file_path = upload_folder / safe_name
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
    """Upload video/audio files."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file types
    allowed_extensions = MEDIA_EXTENSIONS
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
        safe_name = _safe_uploaded_filename(upload_folder, file.filename)
        file_path = upload_folder / safe_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_count += 1
    
    return {
        "message": f"{saved_count} files uploaded",
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


class UrlImportRequest(BaseModel):
    url: str

@router.post("/import-url/{project_id}")
async def import_from_url(
    project_id: int,
    request: UrlImportRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Importă video/audio dintr-un URL web (YouTube, etc)."""
    from app.services.video_processor import download_media_from_url
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    upload_folder = UPLOAD_DIR / f"project_{project_id}"
    
    if "educatieonline.md" in request.url:
        from app.services.educatie_scraper import extract_youtube_urls_from_educatieonline
        try:
            urls = extract_youtube_urls_from_educatieonline(request.url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        if not urls:
            raise HTTPException(status_code=400, detail="No YouTube videos found on page")
            
        successful = 0
        last_error = ""
        last_filename = ""
        for u in urls:
            res = download_media_from_url(u, str(upload_folder))
            if res.get('success'):
                successful += 1
                last_filename = res.get('filename')
            else:
                last_error = res.get('error', "Unknown error")
                
        if successful == 0:
            raise HTTPException(status_code=400, detail=f"Failed to download any video. Last error: {last_error}")
            
        return {
            "message": f"Successfully imported {successful} videos from EducatieOnline",
            "filename": f"{successful} files downloaded (last: {last_filename})",
            "title": f"Bulk import: {successful} videos"
        }
    
    else:
        # Păstrăm fluxul clasic pentru un singur link de YouTube
        result = download_media_from_url(request.url, str(upload_folder))
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=f"Download failed: {result.get('error')}")
            
        return {
            "message": "Download successful",
            "filename": result['filename'],
            "title": result['title']
        }

