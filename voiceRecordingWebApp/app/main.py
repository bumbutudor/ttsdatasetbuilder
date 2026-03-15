import os
import shutil
import csv
import subprocess
import datetime
import zipfile
import io
import glob
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import SessionLocal, Task, engine

# Configurare
DATA_FOLDER = "/data"
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.csv")
LOCK_TIMEOUT_MINUTES = 5

# Read ROOT_PATH from environment for reverse proxy support
root_path = os.getenv("ROOT_PATH", "")

app = FastAPI(root_path=root_path)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/static")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    db = SessionLocal()
    if db.query(Task).count() == 0:
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='|')
                    tasks = []
                    for row in reader:
                        if len(row) >= 3:
                            task = Task(
                                filename=row[0].strip(),
                                original_text=row[1].strip(),
                                normalized_text=row[2].strip(),
                                status="available"
                            )
                            tasks.append(task)
                    db.add_all(tasks)
                    db.commit()
            except Exception as e:
                print(f"Error importing CSV: {e}")
    db.close()

init_db()

def convert_audio(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_audio_duration(file_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def _validate_safe_filename(filename: str) -> str:
    # Prevent path traversal or nested paths
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if any(sep in filename for sep in ("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid filename")
    # Also block parent-dir tricks
    if filename in (".", "..") or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return filename

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/task")
async def get_next_task(username: str, db: Session = Depends(get_db)):
    # 1. Deblochează task-urile vechi (Timeouts)
    expire_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=LOCK_TIMEOUT_MINUTES)
    db.query(Task).filter(Task.status == "locked", Task.locked_at < expire_time).update({"status": "available", "locked_by": None})
    db.commit()

    # 2. Caută un task disponibil ALEATORIU
    # Modificarea critică: order_by(func.random())
    # Astfel, dacă dai skip, e foarte puțin probabil să primești același task înapoi imediat.
    task = db.query(Task).filter(Task.status == "available").order_by(func.random()).first()
    
    if not task:
        # Fallback: verificăm dacă mai sunt task-uri blocate care ar putea expira
        remaining = db.query(Task).filter(Task.status == "locked").count()
        if remaining > 0:
             return JSONResponse({"status": "wait", "message": "Nicio propoziție disponibilă momentan. Încearcă în câteva minute."})
        return JSONResponse({"status": "done", "message": "Toate propozițiile au fost înregistrate! Mulțumim!"})
    
    # 3. Blochează task-ul ("Soft Lock")
    task.status = "locked"
    task.locked_by = username
    task.locked_at = datetime.datetime.utcnow()
    db.commit()
    
    return {
        "id": task.id,
        "text": task.normalized_text,
        "filename": task.filename
    }

@app.post("/api/upload")
async def upload_audio(
    id: int = Form(...),
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    task = db.query(Task).filter(Task.id == id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    temp_filename = f"temp_{id}.webm"
    temp_path = os.path.join(DATA_FOLDER, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        final_path = os.path.join(DATA_FOLDER, task.filename)
        convert_audio(temp_path, final_path)
        
        duration = get_audio_duration(final_path)
        
        if duration < 3.0 or duration > 30.0:
            os.remove(final_path)
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Durata {duration:.1f}s incorectă! (Min 3s, Max 30s)"})

        task.status = "completed"
        task.completed_by = username
        task.completed_at = datetime.datetime.utcnow()
        task.duration_seconds = duration
        db.commit()
        
        return {"status": "success"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/skip")
async def skip_task(id: int = Form(...), db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == id).first()
    if task and task.status == "locked":
        task.status = "available"
        task.locked_by = None
        db.commit()
    return {"status": "skipped"}

# --- ADMIN ROUTES ---

@app.get("/api/admin/stats")
async def get_admin_stats(db: Session = Depends(get_db)):
    total_tasks = db.query(Task).count()
    completed = db.query(Task).filter(Task.status == "completed").count()
    total_duration = db.query(func.sum(Task.duration_seconds)).scalar() or 0
    
    users_stats = db.query(
        Task.completed_by,
        func.count(Task.id),
        func.sum(Task.duration_seconds)
    ).filter(Task.status == "completed").group_by(Task.completed_by).all()
    
    users_data = []
    for u in users_stats:
        users_data.append({
            "username": u[0],
            "count": u[1],
            "minutes": round(u[2] / 60, 2)
        })

    return {
        "total_files": total_tasks,
        "completed": completed,
        "progress_percent": round((completed/total_tasks)*100, 2) if total_tasks > 0 else 0,
        "total_minutes": round(total_duration / 60, 2),
        "users": users_data
    }

@app.get("/api/admin/user_details/{username}")
async def get_user_details(username: str, db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.status == "completed", Task.completed_by == username).all()
    data = []
    for t in tasks:
        data.append({
            "filename": t.filename,
            "text": t.normalized_text,
            "duration": t.duration_seconds
        })
    return data

@app.get("/api/admin/play/{filename}")
async def play_audio(filename: str):
    file_path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTTPException(status_code=404, detail="File not found")

@app.delete("/api/admin/reset_all")
async def reset_all_data(db: Session = Depends(get_db)):
    # 1. Reset DB status
    db.query(Task).update({
        "status": "available",
        "locked_by": None,
        "locked_at": None,
        "completed_by": None,
        "completed_at": None,
        "duration_seconds": 0.0
    })
    db.commit()
    
    # 2. Delete WAV files
    wav_files = glob.glob(os.path.join(DATA_FOLDER, "*.wav"))
    for f in wav_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
            
    return {"status": "reset_complete"}

@app.get("/api/admin/download")
async def download_dataset(db: Session = Depends(get_db)):
    zip_buffer = io.BytesIO()
    completed_tasks = db.query(Task).filter(Task.status == "completed").all()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer, delimiter='|')
        writer.writerow(["filename", "original_text", "normalized_text", "user", "duration"])
        
        for task in completed_tasks:
            writer.writerow([task.filename, task.original_text, task.normalized_text, task.completed_by, task.duration_seconds])
            wav_path = os.path.join(DATA_FOLDER, task.filename)
            if os.path.exists(wav_path):
                zip_file.write(wav_path, task.filename)
        
        zip_file.writestr("dataset_export.csv", csv_buffer.getvalue())

    zip_buffer.seek(0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return StreamingResponse(
        zip_buffer, 
        media_type="application/zip", 
        headers={"Content-Disposition": f"attachment; filename=dataset_{timestamp}.zip"}
    )

# --- ADĂUGARE PENTRU ADMIN DELETE ---

@app.delete("/api/admin/delete_task/{filename}")
async def delete_task_recording(filename: str, db: Session = Depends(get_db)):
    filename = _validate_safe_filename(filename)

    task = db.query(Task).filter(Task.filename == filename).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.status = "available"
    task.completed_by = None
    task.completed_at = None
    task.locked_by = None
    task.locked_at = None
    task.duration_seconds = 0.0
    db.commit()

    file_path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")

    return {"status": "deleted", "filename": filename}


@app.delete("/api/admin/delete_user_all/{username}")
async def delete_user_all_recordings(username: str, db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.completed_by == username, Task.status == "completed").all()
    count = 0

    for task in tasks:
        task.status = "available"
        task.completed_by = None
        task.completed_at = None
        task.locked_by = None
        task.locked_at = None
        task.duration_seconds = 0.0

        file_path = os.path.join(DATA_FOLDER, task.filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        count += 1

    db.commit()
    return {"status": "deleted_all", "count": count, "username": username}