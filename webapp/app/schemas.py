"""Pydantic schemas for request/response validation."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr

from app.models import DatasetType, ProjectStatus


# ============== Auth Schemas ==============

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


# ============== Project Schemas ==============

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: DatasetType


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectResponse(ProjectBase):
    id: int
    status: ProjectStatus
    folder_path: Optional[str]
    total_entries: int
    recorded_entries: int
    estimated_duration_seconds: int
    created_at: datetime
    updated_at: datetime
    owner_id: int

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int


# ============== Dataset Entry Schemas ==============

class DatasetEntryBase(BaseModel):
    wav_filename: str
    original_text: str
    normalized_text: Optional[str] = None


class DatasetEntryCreate(DatasetEntryBase):
    pass


class DatasetEntryResponse(DatasetEntryBase):
    id: int
    has_audio: bool
    duration_seconds: int
    created_at: datetime
    project_id: int

    class Config:
        from_attributes = True


class DatasetEntryListResponse(BaseModel):
    entries: List[DatasetEntryResponse]
    total: int
    page: int
    per_page: int


# ============== Processing Job Schemas ==============

class ProcessingJobResponse(BaseModel):
    id: int
    job_type: str
    status: ProjectStatus
    progress: int
    message: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    project_id: int

    class Config:
        from_attributes = True


# ============== Upload Schemas ==============

class UploadResponse(BaseModel):
    message: str
    job_id: Optional[int] = None
    files_uploaded: int = 0


# ============== Statistics Schemas ==============

class ProjectStats(BaseModel):
    total_entries: int
    recorded_entries: int
    pending_entries: int
    estimated_duration_hours: float
    recorded_duration_hours: float

