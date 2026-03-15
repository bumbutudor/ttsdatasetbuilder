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

# Initialize FastAPI app - WITH root_path to support Nginx reverse proxy
app = FastAPI(
    title="TTS/STT Dataset Builder",
    description="Web application for creating TTS and STT datasets",
    version="2.0.0",
    root_path=root_path
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

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Middleware or helper to inject common variables like root_path
def render_template(name: str, request: Request, **context):
    context["request"] = request
    context["root_path"] = root_path
    return templates.TemplateResponse(name, context)

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
    return render_template("index.html", request)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return render_template("login.html", request)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page."""
    return render_template("register.html", request)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard."""
    return render_template("dashboard.html", request)


@app.get("/files", response_class=HTMLResponse)
async def files_page(request: Request):
    """Uploaded files management page."""
    return render_template("files.html", request)


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    """Generate CSV page."""
    return render_template("generate.html", request)


@app.get("/normalize", response_class=HTMLResponse)
async def normalize_page(request: Request):
    """Normalize text page."""
    return render_template("normalize.html", request)


@app.get("/recorder", response_class=HTMLResponse)
async def recorder_page(request: Request):
    """Voice recorder page."""
    return render_template("recorder.html", request)


@app.get("/video", response_class=HTMLResponse)
async def video_page(request: Request):
    """Video to dataset page."""
    return render_template("video.html", request)


@app.get("/cleanse", response_class=HTMLResponse)
async def cleanse_page(request: Request):
    """Cleanse dataset page."""
    return render_template("cleanse.html", request)


@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    """View dataset page."""
    return render_template("dataset.html", request)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return render_template("settings.html", request)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy"}


# Note: Static files are now served via explicit route above


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
