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
import stt_config
from rich.console import Console

# Try imports
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    WhisperProcessor = None
    WhisperForConditionalGeneration = None

try:
    import torch
except Exception:
    torch = None

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

# Module-level console for consistent logging across functions
console = Console()

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
    Splits audio using Silero VAD to detect natural speech segments.
    Preserves original room tone instead of zero-padding. Falls back
    to the previous librosa-based method if torch/Silero isn't available.
    Returns a list of numpy arrays and the sample rate (44100).
    """
    sr_hq = 44100

    # If torch or silero hub is not available, fall back to librosa-based split
    if torch is None:
        console.print("[yellow]Torch not available — using librosa-based splitting fallback.[/yellow]")
        y, sr = librosa.load(audio_path, sr=sr_hq)
        top_db = 45 if mode == 'TTS' else 30
        intervals = librosa.effects.split(y, top_db=top_db)

        # Merge and chunk similarly to previous behavior
        MIN_SILENCE_DURATION = 0.5
        merged_intervals = []
        if len(intervals) > 0:
            curr_start, curr_end = intervals[0]
            for next_start, next_end in intervals[1:]:
                silence_gap = (next_start - curr_end) / sr
                if silence_gap < MIN_SILENCE_DURATION:
                    curr_end = next_end
                else:
                    merged_intervals.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_intervals.append((curr_start, curr_end))

        chunks = []
        pad_samples = int((0.2 if mode == 'TTS' else 0.5) * sr)
        min_samples = int(min_dur * sr)
        max_samples = int(max_dur * sr)

        current_parts = []
        current_len = 0
        join_pause = np.zeros(int(0.1 * sr))

        for start, end in merged_intervals:
            seg = y[start:end]
            seg_len = len(seg)
            if seg_len > max_samples:
                continue
            added = seg_len + (len(join_pause) if current_parts else 0)
            if current_len + added <= max_samples:
                if current_parts:
                    current_parts.append(join_pause)
                    current_len += len(join_pause)
                current_parts.append(seg)
                current_len += seg_len
            else:
                if current_len >= min_samples:
                    full = np.concatenate(current_parts)
                    full = np.pad(full, (pad_samples, pad_samples), mode='constant')
                    chunks.append(full)
                current_parts = [seg]
                current_len = seg_len

        if current_parts and current_len >= min_samples:
            full = np.concatenate(current_parts)
            full = np.pad(full, (pad_samples, pad_samples), mode='constant')
            chunks.append(full)

        return chunks, sr

    # --- Use Silero VAD ---
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        console.print("[green]Silero VAD loaded — using Silero VAD for voice activity detection.[/green]")
    except Exception as e:
        console.print(f"[yellow]Silero VAD load failed ({e}) — falling back to librosa-based splitting.[/yellow]")
        return split_audio(audio_path, min_dur, max_dur, mode)

    # Silero expects 16k for detection; load for VAD
    try:
        wav_16k = read_audio(audio_path, sampling_rate=16000)
    except Exception:
        # read_audio may fail for some formats; fallback to librosa load+resample
        wav_tmp, _ = librosa.load(audio_path, sr=16000)
        wav_16k = wav_tmp

    # High-quality original for slicing
    wav_hq, sr_hq = librosa.load(audio_path, sr=sr_hq)

    # Get timestamps (start/end in samples at 16k)
    speech_timestamps = get_speech_timestamps(wav_16k, model, sampling_rate=16000, threshold=0.5)

    if not speech_timestamps:
        console.print("[yellow]Silero VAD detected no speech segments; returning empty list.[/yellow]")
        return [], sr_hq

    # Convert timestamps from 16k to sr_hq
    scale = sr_hq / 16000.0
    adjusted = []
    for seg in speech_timestamps:
        start = int(seg['start'] * scale)
        end = int(seg['end'] * scale)
        adjusted.append((start, end))

    # Group segments into chunks trying to preserve natural pauses and not exceed max_dur
    chunks = []
    target_len = int(max_dur * sr_hq)
    min_len = int(min_dur * sr_hq)
    context = int(0.1 * sr_hq)

    cur_start, cur_end = adjusted[0]
    for s, e in adjusted[1:]:
        potential_length = e - cur_start
        if potential_length < target_len:
            cur_end = e
        else:
            safe_start = max(0, cur_start - context)
            safe_end = min(len(wav_hq), cur_end + context)
            seg_audio = wav_hq[safe_start:safe_end]
            if len(seg_audio) >= min_len:
                chunks.append(seg_audio)
            cur_start, cur_end = s, e

    # Add last
    safe_start = max(0, cur_start - context)
    safe_end = min(len(wav_hq), cur_end + context)
    seg_audio = wav_hq[safe_start:safe_end]
    if len(seg_audio) >= min_len:
        chunks.append(seg_audio)

    return chunks, sr_hq

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
    
    # Note: moviepy is only required if there are video files to process (.mp4).
    # Keep Model requirement (pywhispercpp) checked later when loading GGML model.

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
    # Ask user which sample rate to save dataset files at
    console.print("\nAlege frecvența de salvare a fișierelor audio pentru dataset:")
    console.print("1. 16000 Hz (recommended for STT)")
    console.print("2. 44100 Hz (recommended for TTS/high quality)")
    sr_choice = input("Choice (1/2, default 1): ").strip()
    if sr_choice == '2':
        save_sr = 44100
    else:
        save_sr = 16000
    console.print(f"Fișierele vor fi salvate la {save_sr} Hz")
    
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
    whisper_model = None
    hf_processor = None
    hf_model = None
    device = None

    if stt_config.MODEL_SOURCE == 'ggml':
        model_path = os.path.join(app_folder, "models", stt_config.MODEL_NAME)
        if not os.path.exists(model_path):
            console.print(f"[red]Model not found at {model_path}[/red]")
            console.print("Please ensure the model file exists.")
            sys.exit(1)
            
        console.print(f"Loading GGML Whisper model from [cyan]{model_path}[/cyan]...")
        if Model is None:
             console.print("[red]pywhispercpp is not installed![/red]")
             sys.exit(1)
        try:
            # n_threads can be adjusted. 
            whisper_model = Model(model_path, n_threads=6, print_realtime=False, print_progress=False)
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}[/red]")
            sys.exit(1)
            
    elif stt_config.MODEL_SOURCE == 'huggingface':
        model_name = stt_config.MODEL_NAME
        console.print(f"Loading HuggingFace Whisper model: [cyan]{model_name}[/cyan]...")
        
        if WhisperProcessor is None or torch is None:
             console.print("[red]transformers or torch is not installed! Please install them.[/red]")
             sys.exit(1)
             
        try:
            hf_processor = WhisperProcessor.from_pretrained(model_name)
            hf_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hf_model.to(device)
            console.print(f"Model loaded on [green]{device}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load HuggingFace model: {e}[/red]")
            sys.exit(1)
    else:
        console.print(f"[red]Unknown MODEL_SOURCE in config: {stt_config.MODEL_SOURCE}[/red]")
        sys.exit(1)
        
    # 4. Scan media folder (supports .mp4 and .wav)
    console.print("\nPlease enter the path to the media folder (leave empty to use the project's 'videos' folder):")
    in_media_folder = input("Media folder path (default: project/videos): ").strip()
    if in_media_folder:
        media_folder = in_media_folder
    else:
        media_folder = os.path.join(app_folder, "videos")

    if not os.path.exists(media_folder):
        os.mkdir(media_folder)
        console.print(f"[red]Created media folder at {media_folder}. Please put .mp4 or .wav files in it and restart.[/red]")
        sys.exit(0)

    video_files = glob.glob(os.path.join(media_folder, "*.mp4"))
    wav_files = glob.glob(os.path.join(media_folder, "*.wav"))
    media_files = []
    # Keep type info so we know whether to extract
    media_files += [(p, 'mp4') for p in video_files]
    media_files += [(p, 'wav') for p in wav_files]
    console.print(f"Found {len(video_files)} video files and {len(wav_files)} wav files (total {len(media_files)}).")

    if not media_files:
        sys.exit(0)
        
    # CSV Setup
    csv_file_path = os.path.join(project_folder, 'metadata.csv')
    csv_file = open(csv_file_path, 'a', encoding='utf-8', newline='')
    # Using pipe delimiter as per other scripts
    
    valid_count = 0
    temp_full_audio = os.path.join(project_folder, "temp_full.wav")
    
    for file_path, ftype in track(media_files, description="Processing media files..."):
        console.print(f"Processing [cyan]{os.path.basename(file_path)}[/cyan] ({ftype})...")

        audio_source = None
        created_temp = False

        if ftype == 'mp4':
            # Need moviepy to extract
            if VideoFileClip is None:
                console.print("[red]moviepy is required to process .mp4 files. Please install it: pip install moviepy[/red]")
                continue

            # Extract Audio
            if not extract_audio_from_video(file_path, temp_full_audio):
                continue
            audio_source = temp_full_audio
            created_temp = True
        else:
            # WAV file - use source directly
            audio_source = file_path

        # Split Audio
        chunks, sr = split_audio(audio_source, min_dur, max_dur, mode)
        console.print(f"  -> Found {len(chunks)} valid segments.")
        
        for chunk_data in chunks:
            # Save chunk to temp file for transcription
            # We need to save it as 16-bit PCM for the dataset
            
            # Generate filename
            wav_file_name = (str(valid_count) + '.wav').rjust(12, '0')
            wav_path = os.path.join(project_folder, wav_file_name)
            
            # Resample for saving if needed and write chunk
            if save_sr != sr:
                try:
                    to_save = librosa.resample(chunk_data, orig_sr=sr, target_sr=save_sr)
                except Exception as e:
                    console.print(f"[red]Eroare la resampling pentru salvare: {e}[/red]")
                    continue
            else:
                to_save = chunk_data

            sf.write(wav_path, to_save, save_sr, subtype='PCM_16')

            # Prepare chunk for transcription (must be 16k)
            try:
                if save_sr == 16000:
                    chunk_16k = to_save
                else:
                    chunk_16k = librosa.resample(chunk_data, orig_sr=sr, target_sr=16000)
                
                text = ""
                
                if stt_config.MODEL_SOURCE == 'ggml':
                    temp_16k_path = os.path.join(project_folder, "temp_16k.wav")
                    sf.write(temp_16k_path, chunk_16k, 16000, subtype='PCM_16')

                    # pywhispercpp transcribe takes file path
                    # Force Romanian language
                    segments = whisper_model.transcribe(temp_16k_path, language='ro')
                    
                    # Cleanup temp 16k file
                    if os.path.exists(temp_16k_path):
                        os.remove(temp_16k_path)
                    text = "".join([s.text for s in segments])
                    
                elif stt_config.MODEL_SOURCE == 'huggingface':
                    # HF model takes numpy array directly
                    input_features = hf_processor(chunk_16k, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                    # Force language to Romanian and task to transcribe
                    predicted_ids = hf_model.generate(input_features, language="ro", task="transcribe")
                    transcription = hf_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    text = transcription

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
                    
        # Cleanup temp full audio if we created it for extraction
        if created_temp and os.path.exists(temp_full_audio):
            os.remove(temp_full_audio)
            
    csv_file.close()
    console.print(f"\n[green]Done! Generated {valid_count} samples in {project_folder}[/green]")
