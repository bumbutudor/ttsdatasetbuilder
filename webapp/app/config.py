"""Application configuration."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROJECTS_DIR = DATA_DIR / "projects"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/app.db")

# JWT Settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-123!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# AI Settings
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # "ollama" or "openai"
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3:4b")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Whisper Settings (HuggingFace only)
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "iRaduS/whisper-romanian-finetune")

# File upload limits
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB per file

