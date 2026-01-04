"""Project settings routes."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
import json
import os
import logging

from app.database import get_db
from app.auth import get_current_active_user
from app.models import User, Project, ProjectSettings

router = APIRouter(prefix="/api/settings", tags=["Settings"])

# Logger for this module
logger = logging.getLogger(__name__)


class OllamaModelCheck(BaseModel):
    model: str


@router.post("/ollama/check")
async def check_ollama_model(
    request: OllamaModelCheck,
    current_user: User = Depends(get_current_active_user)
):
    """Check if an Ollama model exists locally."""
    try:
        from ollama import Client

        model_name = request.model

        # Connect explicitly to Ollama host (container)
        ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        client = Client(host=ollama_host)

        try:
            local_models = client.list()

            # Handle both dict and object response formats
            models_list = local_models.get('models', []) if isinstance(local_models, dict) else getattr(local_models, 'models', [])

            # Extract model names - handle both dict and Model object
            full_model_names = []
            for m in models_list:
                if isinstance(m, dict):
                    name = m.get('name', '') or m.get('model', '')
                else:
                    # It's a Model object
                    name = getattr(m, 'model', '') or getattr(m, 'name', '')
                if name:
                    full_model_names.append(name)

            # Check if model exists (with or without tag)
            model_base = model_name.split(':')[0]
            exists = model_name in full_model_names or any(
                m == model_name or m.startswith(model_base + ':') or m.split(':')[0] == model_base
                for m in full_model_names
            )

            return {
                "exists": exists,
                "model": model_name,
                "available_models": full_model_names
            }
        except Exception as e:
            return {
                "exists": False,
                "model": model_name,
                "error": f"Could not connect to Ollama: {str(e)} (host={ollama_host})"
            }
            
    except ImportError:
        raise HTTPException(status_code=500, detail="Ollama library not installed")


@router.get("/ollama/pull/{model_name:path}")
async def pull_ollama_model(
    model_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """Pull (download) an Ollama model with streaming progress."""
    try:
        from ollama import Client

        ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        client = Client(host=ollama_host)

        def generate_progress():
            """Generator that yields SSE events with download progress."""
            try:
                logger.info(f"Starting pull for model: {model_name} from {ollama_host}")
                # Use client.pull with stream=True
                stream = client.pull(model_name, stream=True)

                for chunk in stream:
                    # FIX: Handle both dict and Object responses from Ollama lib
                    if hasattr(chunk, 'model_dump'):
                        # Pydantic v2 objects
                        data = chunk.model_dump()
                    elif hasattr(chunk, '__dict__'):
                        # Standard objects
                        data = chunk.__dict__
                    elif isinstance(chunk, dict):
                        # Dictionary
                        data = chunk
                    else:
                        data = {}

                    status = data.get('status', '')
                    total = data.get('total', 0)
                    completed = data.get('completed', 0)

                    progress = 0
                    if total and total > 0:
                        try:
                            progress = int((completed / total) * 100)
                        except Exception:
                            progress = 0

                    event_data = {
                        "status": status,
                        "progress": progress,
                        "completed": completed,
                        "total": total
                    }
                    
                    # Log progress occasionally to Docker logs to debug
                    if progress > 0 and progress % 10 == 0:
                         logger.info(f"Pulling {model_name}: {progress}% - {status}")

                    yield f"data: {json.dumps(event_data)}\n\n"

                # Final success message
                logger.info(f"Model {model_name} pull completed.")
                yield f"data: {json.dumps({'status': 'success', 'progress': 100, 'message': 'Model downloaded successfully!'})}\n\n"

            except Exception as e:
                logger.error(f"Error pulling model: {str(e)}")
                error_msg = str(e)
                yield f"data: {json.dumps({'status': 'error', 'message': error_msg})}\n\n"
        
        return StreamingResponse(
            generate_progress(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except ImportError:
        raise HTTPException(status_code=500, detail="Ollama library not installed")


class SettingsUpdate(BaseModel):
    # Backward compatibility: ignore any fields sent by old clients.
    model_config = ConfigDict(extra="ignore")

    # Text extraction
    min_sentence_length: Optional[int] = 30
    max_sentence_length: Optional[int] = 100
    min_words: Optional[int] = 5
    
    # Vision LLM
    ai_provider: Optional[str] = "ollama"
    ollama_model: Optional[str] = "gemma3:4b"
    openai_model: Optional[str] = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # Recording
    sample_rate: Optional[int] = 44100
    silence_threshold: Optional[int] = 30
    auto_trim: Optional[bool] = True


@router.get("/{project_id}")
async def get_settings(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project settings."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    settings = db.query(ProjectSettings).filter(
        ProjectSettings.project_id == project_id
    ).first()
    
    if not settings:
        # Return defaults
        return SettingsUpdate().dict()

    # Hide deprecated Whisper settings from the API response.
    data = settings.to_dict()
    data.pop("whisper_model", None)
    data.pop("min_segment_duration", None)
    data.pop("max_segment_duration", None)
    return data


@router.post("/{project_id}")
async def save_settings(
    project_id: int,
    settings_data: SettingsUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save project settings."""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    settings = db.query(ProjectSettings).filter(
        ProjectSettings.project_id == project_id
    ).first()
    
    if not settings:
        settings = ProjectSettings(project_id=project_id)
        db.add(settings)
    
    # Update settings
    settings.min_sentence_length = settings_data.min_sentence_length
    settings.max_sentence_length = settings_data.max_sentence_length
    settings.min_words = settings_data.min_words
    settings.ai_provider = settings_data.ai_provider
    settings.ollama_model = settings_data.ollama_model
    settings.openai_model = settings_data.openai_model
    settings.openai_api_key = settings_data.openai_api_key
    settings.sample_rate = settings_data.sample_rate
    settings.silence_threshold = settings_data.silence_threshold
    settings.auto_trim = settings_data.auto_trim
    
    db.commit()
    
    return {"message": "Settings saved"}

