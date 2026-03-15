"""SQLAlchemy database models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum, Boolean
from sqlalchemy.orm import relationship
import enum

from app.database import Base


class DatasetType(str, enum.Enum):
    TTS = "TTS"
    STT = "STT"


class ProjectStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    projects = relationship("Project", back_populates="owner")


class Project(Base):
    """Project model for organizing datasets."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    dataset_type = Column(Enum(DatasetType), nullable=False)
    status = Column(Enum(ProjectStatus), default=ProjectStatus.PENDING)
    folder_path = Column(String(500))
    
    # Statistics
    total_entries = Column(Integer, default=0)
    recorded_entries = Column(Integer, default=0)
    estimated_duration_seconds = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    entries = relationship("DatasetEntry", back_populates="project", cascade="all, delete-orphan")


class DatasetEntry(Base):
    """Individual dataset entry (sentence + audio)."""
    __tablename__ = "dataset_entries"

    id = Column(Integer, primary_key=True, index=True)
    wav_filename = Column(String(100), nullable=False)
    original_text = Column(Text, nullable=False)
    normalized_text = Column(Text)
    has_audio = Column(Boolean, default=False)
    duration_seconds = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="entries")


class ProcessingJob(Base):
    """Background processing job tracking."""
    __tablename__ = "processing_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String(50), nullable=False)  # 'video', 'document', 'normalize'
    status = Column(Enum(ProjectStatus), default=ProjectStatus.PENDING)
    progress = Column(Integer, default=0)  # 0-100
    message = Column(Text)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)


class ProjectSettings(Base):
    """Project-specific settings."""
    __tablename__ = "project_settings"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, unique=True)
    
    # Text extraction settings
    min_sentence_length = Column(Integer, default=30)
    max_sentence_length = Column(Integer, default=100)
    min_words = Column(Integer, default=5)
    
    # Vision LLM settings
    ai_provider = Column(String(50), default="ollama")  # 'ollama', 'openai', 'gemini'
    ollama_model = Column(String(100), default="gemma3:4b")
    openai_model = Column(String(100), default="gpt-4o-mini")
    openai_api_key = Column(String(255))

    gemini_model = Column(String(100), default="gemini-2.0-flash")
    gemini_api_key = Column(String(255))
    
    # Whisper settings
    whisper_model = Column(String(200), default="iRaduS/whisper-romanian-finetune")
    min_segment_duration = Column(Integer, default=3)
    max_segment_duration = Column(Integer, default=10)
    
    # Recording settings
    sample_rate = Column(Integer, default=44100)
    silence_threshold = Column(Integer, default=30)
    auto_trim = Column(Boolean, default=True)
    
    def to_dict(self):
        return {
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "min_words": self.min_words,
            "ai_provider": self.ai_provider,
            "ollama_model": self.ollama_model,
            "openai_model": self.openai_model,
            "openai_api_key": self.openai_api_key,
            "gemini_model": self.gemini_model,
            "gemini_api_key": self.gemini_api_key,
            "whisper_model": self.whisper_model,
            "min_segment_duration": self.min_segment_duration,
            "max_segment_duration": self.max_segment_duration,
            "sample_rate": self.sample_rate,
            "silence_threshold": self.silence_threshold,
            "auto_trim": self.auto_trim
        }

