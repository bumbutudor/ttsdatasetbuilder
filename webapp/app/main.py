"""Main FastAPI application."""
import logging
import os

# Ensure module loggers at INFO are printed to stdout (so container logs show our logger.info)
logging.basicConfig(level=logging.INFO)
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.database import init_db
from app.routes import auth, projects, files, generate, normalize, recorder, video, cleanse, dataset, settings, jobs

# Read ROOT_PATH from environment (set by docker-compose for reverse proxy support)
root_path = os.getenv("ROOT_PATH", "")

# Initialize FastAPI app - DO NOT use root_path for now to debug static files
app = FastAPI(
    title="TTS/STT Dataset Builder",
    description="Web application for creating TTS and STT datasets",
    version="2.0.0"
    # root_path=root_path  # Disabled temporarily
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Log static directory for debugging
logger = logging.getLogger(__name__)
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"STATIC_DIR: {STATIC_DIR}")
logger.info(f"STATIC_DIR exists: {STATIC_DIR.exists()}")
logger.info(f"STATIC_DIR is_dir: {STATIC_DIR.is_dir()}")
if STATIC_DIR.exists():
    logger.info(f"STATIC_DIR contents: {list(STATIC_DIR.iterdir())}")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Explicit static file routes (workaround for mount issues)
@app.api_route("/static/{file_path:path}", methods=["GET", "HEAD"])
async def serve_static(file_path: str):
    """Serve static files."""
    full_path = STATIC_DIR / file_path
    if full_path.exists() and full_path.is_file():
        return FileResponse(full_path)
    return {"detail": "Not Found"}, 404

# Include API routes
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(files.router)
app.include_router(generate.router)
app.include_router(normalize.router)
app.include_router(recorder.router)
app.include_router(video.router)
app.include_router(cleanse.router)
app.include_router(dataset.router)
app.include_router(settings.router)
app.include_router(jobs.router)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()


# ============== Frontend Routes ==============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page - redirects to login or dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page."""
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/files", response_class=HTMLResponse)
async def files_page(request: Request):
    """Uploaded files management page."""
    return templates.TemplateResponse("files.html", {"request": request})


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    """Generate CSV page."""
    return templates.TemplateResponse("generate.html", {"request": request})


@app.get("/normalize", response_class=HTMLResponse)
async def normalize_page(request: Request):
    """Normalize text page."""
    return templates.TemplateResponse("normalize.html", {"request": request})


@app.get("/recorder", response_class=HTMLResponse)
async def recorder_page(request: Request):
    """Voice recorder page."""
    return templates.TemplateResponse("recorder.html", {"request": request})


@app.get("/video", response_class=HTMLResponse)
async def video_page(request: Request):
    """Video to dataset page."""
    return templates.TemplateResponse("video.html", {"request": request})


@app.get("/cleanse", response_class=HTMLResponse)
async def cleanse_page(request: Request):
    """Cleanse dataset page."""
    return templates.TemplateResponse("cleanse.html", {"request": request})


@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    """View dataset page."""
    return templates.TemplateResponse("dataset.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse("settings.html", {"request": request})


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy"}


# Note: Static files are now served via explicit route above


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
