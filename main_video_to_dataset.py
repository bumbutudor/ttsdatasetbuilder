from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress
import os
import glob
import sys
from datetime import datetime, timedelta
import numpy as np
import soundfile as sf
import librosa
import csv
import shutil

# Try imports
try:
    # Try importing from moviepy.editor (v1.x)
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        # Try importing directly from moviepy (v2.x)
        from moviepy import VideoFileClip
    except ImportError:
        VideoFileClip = None
        print("Error: 'moviepy' library is required. Please install it: pip install moviepy")

try:
    from pywhispercpp.model import Model
except ImportError:
    Model = None
    print("Error: 'pywhispercpp' library is required for GGML models. Please install it: pip install pywhispercpp")

def extract_audio_from_video(video_path, temp_audio_path):
    """Extracts audio from video using moviepy."""
    try:
        video = VideoFileClip(video_path)
        # Extract audio at 44100Hz, PCM 16-bit
        # moviepy writes using ffmpeg. codec='pcm_s16le' ensures 16-bit PCM.
        video.audio.write_audiofile(temp_audio_path, fps=44100, nbytes=2, codec='pcm_s16le', logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False

def split_audio(audio_path, min_dur, max_dur, mode):
    """
    Splits audio into chunks based on silence and duration constraints.
    Returns a list of (audio_data, sample_rate) tuples.
    """
    # Load with librosa (converts to float32 by default, which is fine for processing)
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Detect non-silent intervals
    # top_db: The threshold (in decibels) below reference to consider as silence
    # Increased to 45/40 to avoid cutting words with dynamic volume (standard is 60, but videos might be noisy)
    top_db = 45 if mode == 'TTS' else 40 
    intervals = librosa.effects.split(y, top_db=top_db)
    
    chunks = []
    current_chunk = []
    current_samples = 0
    
    min_samples = int(min_dur * sr)
    max_samples = int(max_dur * sr)
    
    for start, end in intervals:
        segment = y[start:end]
        seg_len = len(segment)
        
        # If a single segment is longer than max_dur, we have to skip it or split it arbitrarily.
        # For dataset quality, skipping is often safer unless we have a smart splitter.
        if seg_len > max_samples:
            continue 
            
        if current_samples + seg_len <= max_samples:
            # Append to current chunk
            # Add a small silence (0.1s) to separate words/phrases if merging
            if len(current_chunk) > 0:
                 pause_len = int(0.1 * sr)
                 pause = np.zeros(pause_len)
                 current_chunk.append(pause)
                 current_samples += pause_len
            
            current_chunk.append(segment)
            current_samples += seg_len
        else:
            # Current chunk is full-ish. Check if it meets min duration
            if current_samples >= min_samples:
                full_chunk = np.concatenate(current_chunk)
                chunks.append(full_chunk)
            
            # Start new chunk with current segment
            current_chunk = [segment]
            current_samples = seg_len
            
    # Last chunk
    if current_samples >= min_samples and current_chunk:
        full_chunk = np.concatenate(current_chunk)
        chunks.append(full_chunk)
        
    return chunks, sr

def clean_text(text):
    """Basic cleanup of whisper output."""
    text = text.strip()
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    return text

if __name__ == '__main__':
    console = Console()
    
    table = Table()
    table.add_column("Video to Dataset Generator (Whisper)", style="cyan")
    table.add_row("Extracts audio from videos, splits it, and transcribes with Whisper.")
    table.add_row("Generates a dataset compatible with TTS/STT training.")
    console.print(table)
    
    # Check for libraries
    if VideoFileClip is None:
        console.print("[red]Critical: moviepy not installed or failed to import.[/red]")
        console.print("Please run: [yellow]pip install moviepy[/yellow]")
        sys.exit(1)
            
    if Model is None:
        console.print("[red]Critical: pywhispercpp not installed or failed to import.[/red]")
        console.print("Please run: [yellow]pip install pywhispercpp[/yellow]")
        sys.exit(1)

    # 1. Select Mode
    console.print("\nSelect dataset type:")
    console.print("1. [cyan]TTS[/cyan] (Text-to-Speech) - 3-10s segments.")
    console.print("2. [green]STT[/green] (Speech-to-Text) - 3-30s segments.")
    mode_input = input("Choice (1/2): ").strip()
    
    if mode_input == '2':
        mode = 'STT'
        folder_prefix = 'project_VIDEO_STT_'
        min_dur = 3
        max_dur = 30
    else:
        mode = 'TTS'
        folder_prefix = 'project_VIDEO_TTS_'
        min_dur = 3
        max_dur = 10
        
    console.print(f"Selected mode: [bold]{mode}[/bold]")
    
    # 2. Setup Project Folder
    app_folder = os.path.dirname(os.path.realpath(__file__))
    now = datetime.now()
    project_folder = os.path.join(app_folder, folder_prefix + now.strftime("%d%m%Y_%H%M%S"))
    
    console.print(f"Please select a [red]project folder[/red] (default [i]{project_folder}[/i]).")
    in_project_folder = input().strip()
    if not in_project_folder:
        in_project_folder = project_folder
    project_folder = in_project_folder
    
    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    console.print(f"Project folder is [yellow]{project_folder}[/yellow]")
    
    # 3. Load Whisper Model
    model_path = os.path.join(app_folder, "models", "ggml-whisper-medium-romanian.bin")
    if not os.path.exists(model_path):
        console.print(f"[red]Model not found at {model_path}[/red]")
        console.print("Please ensure the model file exists.")
        sys.exit(1)
        
    console.print(f"Loading Whisper model from [cyan]{model_path}[/cyan]...")
    try:
        # n_threads can be adjusted. 
        whisper_model = Model(model_path, n_threads=6, print_realtime=False, print_progress=False)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        sys.exit(1)
        
    # 4. Scan Videos
    videos_folder = os.path.join(app_folder, "videos")
    if not os.path.exists(videos_folder):
        os.mkdir(videos_folder)
        console.print(f"[red]Created 'videos' folder. Please put .mp4 files in it and restart.[/red]")
        sys.exit(0)
        
    video_files = glob.glob(os.path.join(videos_folder, "*.mp4"))
    console.print(f"Found {len(video_files)} video files.")
    
    if not video_files:
        sys.exit(0)
        
    # CSV Setup
    csv_file_path = os.path.join(project_folder, 'metadata.csv')
    csv_file = open(csv_file_path, 'a', encoding='utf-8', newline='')
    # Using pipe delimiter as per other scripts
    
    valid_count = 0
    temp_full_audio = os.path.join(project_folder, "temp_full.wav")
    
    for video_file in track(video_files, description="Processing videos..."):
        console.print(f"Processing [cyan]{os.path.basename(video_file)}[/cyan]...")
        
        # Extract Audio
        if not extract_audio_from_video(video_file, temp_full_audio):
            continue
            
        # Split Audio
        chunks, sr = split_audio(temp_full_audio, min_dur, max_dur, mode)
        console.print(f"  -> Found {len(chunks)} valid segments.")
        
        for chunk_data in chunks:
            # Save chunk to temp file for transcription
            # We need to save it as 16-bit PCM for the dataset
            
            # Generate filename
            wav_file_name = (str(valid_count) + '.wav').rjust(12, '0')
            wav_path = os.path.join(project_folder, wav_file_name)
            
            # Write chunk (44100 Hz for dataset)
            sf.write(wav_path, chunk_data, sr, subtype='PCM_16')
            
            # Transcribe (requires 16000 Hz)
            try:
                # Resample to 16k for Whisper
                # librosa.resample expects float input, chunk_data is likely float from librosa.load
                chunk_16k = librosa.resample(chunk_data, orig_sr=sr, target_sr=16000)
                temp_16k_path = os.path.join(project_folder, "temp_16k.wav")
                sf.write(temp_16k_path, chunk_16k, 16000, subtype='PCM_16')

                # pywhispercpp transcribe takes file path
                # Force Romanian language
                segments = whisper_model.transcribe(temp_16k_path, language='ro')
                
                # Cleanup temp 16k file
                if os.path.exists(temp_16k_path):
                    os.remove(temp_16k_path)
                text = "".join([s.text for s in segments])
                text = clean_text(text)
                
                if not text or len(text) < 2:
                    # Empty transcription, delete file
                    os.remove(wav_path)
                    continue
                    
                # Write to CSV: filename|text|text
                # Using pipe delimiter manually to match format
                csv_file.write(f"{wav_file_name}|{text}|{text}\n")
                csv_file.flush()
                valid_count += 1
                
            except Exception as e:
                console.print(f"[red]Error transcribing segment: {e}[/red]")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                    
        # Cleanup temp full audio
        if os.path.exists(temp_full_audio):
            os.remove(temp_full_audio)
            
    csv_file.close()
    console.print(f"\n[green]Done! Generated {valid_count} samples in {project_folder}[/green]")
