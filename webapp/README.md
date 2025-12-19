# TTS/STT Dataset Builder - Web Application

A collaborative web platform for creating Text-to-Speech (TTS) and Speech-to-Text (STT) datasets, designed for research teams working with Romanian language models.

## Features

- 👥 **Multi-user Support**: Registration and authentication for team collaboration
- 📄 **Document Processing**: Upload PDF and TXT files, automatically extract and split sentences
- 🎥 **Video Processing**: Upload videos, extract audio, split on silence, and transcribe with Whisper
- 🤖 **Vision AI**: Use OpenAI or Ollama vision models for complex PDFs (formulas, tables)
- 📊 **Dataset Management**: View, download, and clean datasets
- 🐳 **Docker Support**: Easy deployment with Docker Compose

## Quick Start with Docker

### 1. Clone and Configure

```bash
cd webapp

# Copy environment configuration
cp env.example.txt .env

# Edit .env with your settings (especially OPENAI_API_KEY if using OpenAI)
```

### 2. Build and Run

```bash
# Build and start the application
docker-compose up -d --build

# View logs
docker-compose logs -f app
```

### 3. Access the Application

Open your browser and navigate to: **http://localhost:8000**

1. Register a new account
2. Create a project (TTS or STT)
3. Upload documents or videos
4. View and download your dataset

## Local Development

### Prerequisites

- Python 3.11+
- FFmpeg (for video processing)
- CUDA (optional, for GPU-accelerated Whisper)

### Installation

```bash
cd webapp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key (change in production!) | `your-super-secret-key...` |
| `DATABASE_URL` | Database connection string | `sqlite:///./data/app.db` |
| `AI_PROVIDER` | Vision AI provider: `openai` or `ollama` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL_NAME` | OpenAI model for vision | `gpt-4o-mini` |
| `OLLAMA_MODEL_NAME` | Ollama model name | `gemma3:4b` |
| `WHISPER_MODEL_NAME` | HuggingFace Whisper model | `iRaduS/whisper-romanian-finetune` |

### Dataset Types

#### TTS (Text-to-Speech)
- Strict sentence filtering (30-100 characters)
- Short audio segments (3-10 seconds)
- Text normalization: numbers to words, abbreviation expansion

#### STT (Speech-to-Text)
- Relaxed sentence filtering (30-220 characters)
- Longer audio segments (3-30 seconds)
- Text normalization: preserve numbers, fix punctuation

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/me` - Get current user info

### Projects
- `GET /api/projects` - List all projects
- `POST /api/projects` - Create new project
- `GET /api/projects/{id}` - Get project details
- `DELETE /api/projects/{id}` - Delete project
- `GET /api/projects/{id}/download` - Download dataset as ZIP
- `POST /api/projects/{id}/cleanse` - Remove entries without audio

### Upload & Processing
- `POST /api/upload/documents/{project_id}` - Upload PDF/TXT files
- `POST /api/upload/videos/{project_id}` - Upload video files
- `GET /api/upload/jobs/{job_id}` - Get processing job status

## Architecture

```
webapp/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration
│   ├── database.py       # SQLAlchemy setup
│   ├── models.py         # Database models
│   ├── schemas.py        # Pydantic schemas
│   ├── auth.py           # Authentication utilities
│   ├── routes/           # API endpoints
│   │   ├── auth.py       # Auth routes
│   │   ├── projects.py   # Project routes
│   │   └── upload.py     # Upload routes
│   ├── services/         # Business logic
│   │   ├── document_processor.py
│   │   ├── video_processor.py
│   │   ├── vision_processor.py
│   │   └── text_normalizer.py
│   ├── static/           # CSS, JS
│   └── templates/        # Jinja2 templates
├── data/                 # Persistent storage
│   ├── uploads/          # Uploaded files
│   └── projects/         # Project datasets
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Whisper Models

The application uses HuggingFace Transformers for Whisper (no GGML). Available Romanian models:

| Model | Size | Notes |
|-------|------|-------|
| `iRaduS/whisper-romanian-finetune` | Medium | Recommended, good balance |
| `TransferRapid/whisper-large-v3-turbo_ro` | Large | Best quality, slower |
| `gigant/whisper-medium-romanian` | Medium | Alternative option |
| `readerbench/whisper-ro` | Medium | Academic |

## Production Deployment

### Security Checklist

1. ✅ Change `SECRET_KEY` to a strong random value
2. ✅ Use HTTPS (configure reverse proxy like nginx)
3. ✅ Set up proper firewall rules
4. ✅ Use PostgreSQL instead of SQLite for multiple users
5. ✅ Configure backup for the `data/` directory

### Docker Production Settings

```yaml
# docker-compose.prod.yml
services:
  app:
    restart: always
    environment:
      - SECRET_KEY=${SECRET_KEY}  # From .env, not in compose file
    deploy:
      resources:
        limits:
          memory: 8G
```

### With GPU Support

Uncomment the GPU sections in `docker-compose.yml` for NVIDIA GPU acceleration.

## Troubleshooting

### "Out of memory" during video processing
- Reduce video resolution before upload
- Use smaller Whisper model
- Add more system RAM or enable GPU

### "Model not found" for Whisper
- First run downloads the model (~1-2GB)
- Ensure internet connectivity
- Check HuggingFace model name is correct

### Slow PDF processing
- Use Vision AI only for complex documents
- Simple PDFs process faster with standard text extraction

## License

MIT License - See LICENSE file for details.

## Credits

Based on the original TTS Dataset Builder scripts, adapted for web deployment.

