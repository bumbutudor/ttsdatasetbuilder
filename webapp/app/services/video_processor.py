"""Video processing service - extract audio, split, and transcribe."""
import os
import shutil
import tempfile
import logging
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

def extract_audio_from_video(video_path, temp_audio_path):
    """Extracts audio from video using moviepy."""
    if VideoFileClip is None:
        logger.error("MoviePy library not loaded!")
        raise RuntimeError("MoviePy is not installed.")
        
    try:
        logger.info(f"Extracting audio from {video_path} to {temp_audio_path}")
        
        # Load Video
        video = VideoFileClip(video_path)
        
        # Check Audio Track
        if video.audio is None:
            logger.error(f"Video {video_path} has no audio track!")
            video.close()
            return False

        # Extract Audio
        # FIX: Removed 'verbose=True' causing TypeError
        video.audio.write_audiofile(
            temp_audio_path, 
            fps=44100, 
            nbytes=2, 
            codec='pcm_s16le', 
            logger=None
        )
        video.close()
        
        # Validate Output
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            logger.error("Extracted audio file is missing or empty!")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def split_audio_logic(audio_path, min_dur, max_dur, min_silence, padding, threshold):
    """
    Core splitting logic ported EXACTLY from your working script.
    """
    # Load with librosa at 44100 (high quality for dataset storage)
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Detect non-silent intervals
    intervals = librosa.effects.split(y, top_db=threshold)
    
    # 1. Merge intervals that are too close
    merged_intervals = []
    if len(intervals) > 0:
        curr_start, curr_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            silence_gap = (next_start - curr_end) / sr
            if silence_gap < min_silence:
                # Merge with previous
                curr_end = next_end
            else:
                # Save current and start new
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_intervals.append((curr_start, curr_end))
    
    chunks = []
    current_chunk_parts = []
    current_len = 0
    
    min_samples = int(min_dur * sr)
    max_samples = int(max_dur * sr)
    pad_samples = int(padding * sr)
    
    # Small pause to insert between joined segments
    join_pause_samples = int(0.1 * sr) 
    join_pause = np.zeros(join_pause_samples)
    
    for start, end in merged_intervals:
        segment = y[start:end]
        seg_len = len(segment)
        
        # Skip overly long single segments
        if seg_len > max_samples:
            continue 
            
        # Calculate potential length
        added_len = seg_len
        if current_chunk_parts:
            added_len += join_pause_samples

        if current_len + added_len <= max_samples:
            # Append to current chunk
            if current_chunk_parts:
                 current_chunk_parts.append(join_pause)
                 current_len += join_pause_samples
            
            current_chunk_parts.append(segment)
            current_len += seg_len
        else:
            # Current chunk is full-ish. Check if it meets min duration
            if current_len >= min_samples:
                full_audio = np.concatenate(current_chunk_parts)
                # Add padding at start/end
                full_audio = np.pad(full_audio, (pad_samples, pad_samples), mode='constant')
                chunks.append(full_audio)
            
            # Start new chunk with current segment
            current_chunk_parts = [segment]
            current_len = seg_len
            
    # Last chunk
    if current_chunk_parts and current_len >= min_samples:
        full_audio = np.concatenate(current_chunk_parts)
        full_audio = np.pad(full_audio, (pad_samples, pad_samples), mode='constant')
        chunks.append(full_audio)
        
    return chunks, sr

def split_audio_staging(
    video_paths: List[str],
    staging_folder: str,
    min_dur: float,
    max_dur: float,
    min_silence: float,
    padding: float,
    threshold: int,
    progress_callback: Callable,
    check_cancel: Callable
) -> List[Dict]:
    """
    Step 1: Extract and Split audio into a staging folder using the robust logic.
    """
    os.makedirs(staging_folder, exist_ok=True)
    segments_info = []
    
    total_videos = len(video_paths)
    
    for v_idx, video_path in enumerate(video_paths):
        if check_cancel(): break
        
        video_name = Path(video_path).stem
        progress_callback(int((v_idx / total_videos) * 100), f"Processing {video_name}...")
        
        # Temp file for full audio
        # Use simple name to avoid character issues in temp paths if possible
        temp_audio_name = f"temp_{v_idx}.wav"
        temp_audio_path = os.path.join(staging_folder, temp_audio_name)
            
        try:
            # 1. Extract Audio
            success = extract_audio_from_video(video_path, temp_audio_path)
            if not success:
                logger.error(f"Failed to extract audio from {video_name}")
                continue
            
            if check_cancel(): break

            # 2. Split Audio using the logic from your script
            chunks, sr = split_audio_logic(
                temp_audio_path, 
                min_dur, max_dur, 
                min_silence, padding, threshold
            )
            
            logger.info(f"Video {video_name}: Found {len(chunks)} segments")
            
            # 3. Save Chunks
            for i, chunk_data in enumerate(chunks):
                if check_cancel(): break
                
                seg_filename = f"{video_name}_seg_{i:04d}.wav"
                seg_path = os.path.join(staging_folder, seg_filename)
                
                # Save as 16-bit PCM (standard for datasets)
                sf.write(seg_path, chunk_data, sr, subtype='PCM_16')
                
                segments_info.append({
                    "filename": seg_filename,
                    "path": seg_path,
                    "duration": round(len(chunk_data)/sr, 2)
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