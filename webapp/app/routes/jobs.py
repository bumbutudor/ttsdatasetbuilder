"""Processing jobs routes."""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, ProcessingJob
from app.schemas import ProcessingJobResponse

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=ProcessingJobResponse)
async def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get processing job status."""
    job = db.query(ProcessingJob).filter(
        ProcessingJob.id == job_id,
        ProcessingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job


@router.get("", response_model=List[ProcessingJobResponse])
async def list_jobs(
    project_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all processing jobs for the current user."""
    query = db.query(ProcessingJob).filter(
        ProcessingJob.user_id == current_user.id
    )
    
    if project_id:
        query = query.filter(ProcessingJob.project_id == project_id)
    
    jobs = query.order_by(ProcessingJob.created_at.desc()).limit(50).all()
    
    return jobs

