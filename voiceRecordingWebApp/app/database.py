from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Database stored in mounted /data folder for persistence across container restarts
SQLALCHEMY_DATABASE_URL = "sqlite:////data/stt_tasks.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    original_text = Column(String)
    normalized_text = Column(String) # Coloana 3
    
    # Status: available, locked, completed, skipped
    status = Column(String, default="available", index=True)
    
    locked_by = Column(String, nullable=True)
    locked_at = Column(DateTime, nullable=True)
    
    completed_by = Column(String, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, default=0.0)

Base.metadata.create_all(bind=engine)