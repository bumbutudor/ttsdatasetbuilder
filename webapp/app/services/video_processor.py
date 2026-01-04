"""Video processing service - extract audio, split, and transcribe."""
import os
import shutil
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from app.config import WHISPER_MODEL_NAME

logger = logging.getLogger(__name__)

# --- Imports compatibili cu scriptul tau ---
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        VideoFileClip = None
        logger.warning("moviepy not found")

# Lazy load globals
_whisper_processor = None
_whisper_model = None
_device = None
_current_model_name = None
_silero_model = None
_silero_utils = None

def load_silero_model():
    """Load Silero VAD model from torch hub."""
    global _silero_model, _silero_utils
    if _silero_model is not None:
        return _silero_model, _silero_utils
        
    try:
        logger.info("Loading Silero VAD model...")
        # Force reload=False to use cache if available
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False,
                                      trust_repo=True)
        _silero_model = model
        _silero_utils = utils
        return model, utils
    except Exception as e:
        logger.error(f"Failed to load Silero VAD: {e}")
        raise RuntimeError(f"Failed to load Silero VAD: {e}")

def check_model_exists(model_name: str) -> bool:
    """Check if a HuggingFace model exists locally."""
    try:
        from transformers import AutoModel
        # Încercăm să încărcăm configurația locală fără a descărca
        AutoModel.from_pretrained(model_name, local_files_only=True)
        return True
    except Exception:
        return False

def download_model_task(model_name: str, progress_callback: Callable):
    """Background task to download model."""
    try:
        progress_callback(10, "Initializing download...")
        progress_callback(30, "Downloading model weights (this may take a while)...")
        
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        progress_callback(100, "Download complete!")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise e

def _load_whisper_model(model_name: str = None):
    """Lazy load Whisper model."""
    global _whisper_processor, _whisper_model, _device, _current_model_name
    
    model_name = model_name or WHISPER_MODEL_NAME
    
    if _current_model_name == model_name and _whisper_model is not None:
        return _whisper_processor, _whisper_model, _device
    
    logger.info(f"Loading Whisper model: {model_name}...")
    try:
        _whisper_processor = WhisperProcessor.from_pretrained(model_name)
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_model.to(_device)
        _current_model_name = model_name
        logger.info(f"Model loaded on {_device}")
        return _whisper_processor, _whisper_model, _device
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}")

def convert_audio_for_whisper(input_path: str, output_path: str) -> bool:
    """
    Convertește orice input media în WAV, 16000Hz, Mono, PCM 16-bit.
    Folosește FFmpeg direct via subprocess pentru precizie.
    """
    try:
        command = [
            'ffmpeg', '-y',          # Overwrite yes
            '-i', input_path,        # Input
            '-ar', '16000',          # Sample rate 16kHz
            '-ac', '1',              # Mono
            '-c:a', 'pcm_s16le',     # Codec 16-bit PCM
            output_path
        ]
        
        # Rulăm comanda, ascundem output-ul (stderr) dacă nu e eroare
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return False

def download_media_from_url(url: str, output_folder: str) -> dict:
    """Descarcă video/audio dintr-un URL folosind yt-dlp."""
    import yt_dlp
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Configurare pentru a descărca cel mai bun format și a-l salva cu titlul original
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': f'{output_folder}/%(title)s.%(ext)s',
        'restrictfilenames': True,  # Evită caractere speciale în nume
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            # Trunchiere nume fișier (stem <= 25) + evitare coliziuni
            try:
                folder = Path(output_folder)
                original_path = Path(filename)
                ext = original_path.suffix

                stem = original_path.stem
                safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))
                safe_stem = (safe_stem or "file")[:25]

                candidate = folder / f"{safe_stem}{ext}"
                if candidate.exists():
                    counter = 1
                    while True:
                        suffix = f"_{counter}"
                        trimmed = safe_stem[: max(1, 25 - len(suffix))]
                        candidate = folder / f"{trimmed}{suffix}{ext}"
                        if not candidate.exists():
                            break
                        counter += 1

                if original_path.resolve() != candidate.resolve():
                    os.replace(str(original_path), str(candidate))
                    filename = str(candidate)
            except Exception as e:
                logger.warning(f"Could not truncate downloaded filename: {e}")

            # Returnăm doar numele fișierului, nu calea completă
            return {
                "success": True, 
                "filename": os.path.basename(filename),
                "title": info.get('title', 'Unknown')
            }
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return {"success": False, "error": str(e)}

def prepare_media_source(media_path, temp_audio_path):
    """
    Pregătește sursa pentru procesare. 
    Dacă e video -> extrage audio convertit.
    Dacă e audio -> convertește la formatul Whisper.
    """
    # Indiferent dacă e video sau audio, FFmpeg se descurcă să extragă/convertească stream-ul audio
    # folosind aceeași comandă definită mai sus.
    logger.info(f"Processing media source: {media_path} -> {temp_audio_path}")
    return convert_audio_for_whisper(media_path, temp_audio_path)

def process_file_with_silero(file_path, model, utils, min_dur, max_dur, target_sr=44100):
    """
    Procesează un singur fișier audio folosind Silero VAD.
    Adaptat din scriptul split_audio_silero_vad.py
    """
    # Extragem utilitarele
    (get_speech_timestamps, _, _, _, _) = utils
    
    # --- PASUL 1: Încărcare pentru VAD (16000 Hz) ---
    try:
        # Folosim librosa pentru a citi fișierul la 16k
        wav_np, _ = librosa.load(file_path, sr=16000, mono=True)
        # Convertim în tensor Torch
        wav_16k = torch.from_numpy(wav_np)
    except Exception as e:
        logger.error(f"Error reading audio (16k) {os.path.basename(file_path)}: {e}")
        return []

    # --- PASUL 2: Detectare segmente ---
    try:
        # threshold=0.5 (sensibilitate standard)
        # min_speech_duration_ms=250 (ignorăm zgomotele foarte scurte)
        speech_timestamps = get_speech_timestamps(wav_16k, model, sampling_rate=16000, threshold=0.5, min_speech_duration_ms=250)
    except Exception as e:
         logger.error(f"Error executing VAD on {os.path.basename(file_path)}: {e}")
         return []
    
    if not speech_timestamps:
        return []

    # --- PASUL 3: Încărcare High Quality pentru tăiere ---
    try:
        wav_hq, _ = librosa.load(file_path, sr=target_sr)
    except Exception as e:
        logger.error(f"Error reading audio (HQ) {os.path.basename(file_path)}: {e}")
        return []

    # --- PASUL 4: Extragere și Grupare (Corecție Durată) ---
    scale_factor = target_sr / 16000.0
    final_chunks = []
    
    min_samples = int(min_dur * target_sr)
    max_samples = int(max_dur * target_sr)
    
    # Conversie timestamp-uri la sample rate-ul țintă
    adjusted_timestamps = []
    for seg in speech_timestamps:
        start = int(seg['start'] * scale_factor)
        end = int(seg['end'] * scale_factor)
        adjusted_timestamps.append((start, end))

    if not adjusted_timestamps:
        return []

    current_chunk_start = adjusted_timestamps[0][0]
    current_chunk_end = adjusted_timestamps[0][1]
    
    # Buffer mic (0.1s) la capete pentru naturalețe
    context_samples = int(0.1 * target_sr)

    for i in range(1, len(adjusted_timestamps)):
        next_start, next_end = adjusted_timestamps[i]
        
        # Verificăm dacă adăugarea segmentului următor depășește MAX_DUR
        if (next_end - current_chunk_start) < max_samples:
            # Dacă nu depășește, unim segmentele
            current_chunk_end = next_end
        else:
            # Dacă depășește, salvăm ce am acumulat până acum
            s = max(0, current_chunk_start - context_samples)
            e = min(len(wav_hq), current_chunk_end + context_samples)
            segment = wav_hq[s:e]
            
            # Verificăm dacă segmentul rezultat respectă MIN_DUR
            if len(segment) >= min_samples:
                final_chunks.append(segment)
            
            # Resetăm acumularea începând cu segmentul curent
            current_chunk_start = next_start
            current_chunk_end = next_end
            
    # Adăugăm ultimul segment rămas
    s = max(0, current_chunk_start - context_samples)
    e = min(len(wav_hq), current_chunk_end + context_samples)
    segment = wav_hq[s:e]
    if len(segment) >= min_samples:
        final_chunks.append(segment)

    return final_chunks

def split_audio_staging(
    video_paths: List[str],
    staging_folder: str,
    dataset_type: str, # 'tts' or 'stt'
    progress_callback: Callable,
    check_cancel: Callable
) -> List[Dict]:
    """
    Step 1: Extract and Split audio into a staging folder using Silero VAD.
    """
    os.makedirs(staging_folder, exist_ok=True)
    segments_info = []
    
    # Configurare durate bazată pe tipul dataset-ului
    if dataset_type == 'tts':
        min_dur = 3.0
        max_dur = 10.0
    else: # stt
        min_dur = 3.0
        max_dur = 30.0
        
    # Încărcare model Silero
    try:
        model, utils = load_silero_model()
    except Exception as e:
        logger.error(f"Could not load Silero model: {e}")
        raise e
    
    total_videos = len(video_paths)
    
    for v_idx, video_path in enumerate(video_paths):
        if check_cancel(): break
        
        video_name = Path(video_path).stem
        progress_callback(int((v_idx / total_videos) * 100), f"Processing {video_name}...")
        
        # Temp file for full audio
        temp_audio_name = f"temp_{v_idx}.wav"
        temp_audio_path = os.path.join(staging_folder, temp_audio_name)
            
        try:
            # 1. Prepare Audio (Extract & Convert standard)
            success = prepare_media_source(video_path, temp_audio_path)
            
            if not success:
                logger.error(f"Failed to process media {video_name}")
                continue
            
            if check_cancel(): break

            # 2. Split Audio using Silero VAD
            # Folosim 44100Hz pentru calitate maximă la salvare
            chunks = process_file_with_silero(
                temp_audio_path, 
                model, utils, 
                min_dur, max_dur, 
                target_sr=44100
            )
            
            logger.info(f"Video {video_name}: Found {len(chunks)} segments")
            
            # 3. Save Chunks
            for i, chunk_data in enumerate(chunks):
                if check_cancel(): break
                
                seg_filename = f"{video_name}_seg_{i:04d}.wav"
                seg_path = os.path.join(staging_folder, seg_filename)
                
                # Save as 16-bit PCM
                sf.write(seg_path, chunk_data, 44100, subtype='PCM_16')
                
                segments_info.append({
                    "filename": seg_filename,
                    "path": seg_path,
                    "duration": round(len(chunk_data) / 44100, 2)
                })
                
        except Exception as e:
            logger.error(f"Error splitting {video_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    progress_callback(100, f"Splitting complete. Found {len(segments_info)} segments.")
    return segments_info

def transcribe_staging(
    staging_folder: str,
    files_to_transcribe: List[str],
    model_name: str,
    progress_callback: Callable,
    check_cancel: Callable
) -> List[Dict]:
    """
    Step 2: Transcribe specific files from staging.
    """
    # Load model
    processor, model, device = _load_whisper_model(model_name)
    results = []
    
    total = len(files_to_transcribe)
    
    for idx, filename in enumerate(files_to_transcribe):
        if check_cancel(): break
        
        file_path = os.path.join(staging_folder, filename)
        if not os.path.exists(file_path):
            continue
            
        progress_callback(int((idx / total) * 100), f"Transcribing {filename}...")
        
        try:
            # Load audio for Whisper (must be 16000 Hz)
            # librosa handles resampling automatically here
            y, _ = librosa.load(file_path, sr=16000)
            
            # Prepare inputs
            input_features = processor(
                y, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # Generate
            with torch.no_grad():
                # Force Romanian language
                try:
                    if hasattr(processor, "get_decoder_prompt_ids"):
                        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ro", task="transcribe")
                    elif hasattr(processor.tokenizer, "get_decoder_prompt_ids"):
                         forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="ro", task="transcribe")
                    else:
                        forced_decoder_ids = None

                    predicted_ids = model.generate(
                        input_features, 
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=256
                    )
                except Exception:
                    predicted_ids = model.generate(input_features, max_new_tokens=256)
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Cleanup text
            text = transcription.strip().replace('|', '').replace('\n', ' ')
            text = " ".join(text.split())
            
            if len(text) > 1:
                results.append({
                    "filename": filename,
                    "text": text,
                    "path": file_path
                })
            
        except Exception as e:
            logger.error(f"Error transcribing {filename}: {e}")
            results.append({
                "filename": filename,
                "text": "[ERROR]",
                "path": file_path
            })
            
    progress_callback(100, "Transcription complete")
    return results