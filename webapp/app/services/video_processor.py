"""Video processing service - extract audio, split, and transcribe."""
import os
import csv
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
import soundfile as sf
import librosa

from app.config import WHISPER_MODEL_NAME


# Lazy load heavy dependencies
_whisper_processor = None
_whisper_model = None
_device = None


def _load_whisper_model():
    """Lazy load Whisper model from HuggingFace."""
    global _whisper_processor, _whisper_model, _device
    
    if _whisper_model is not None:
        return _whisper_processor, _whisper_model, _device
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        
        _whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_model.to(_device)
        
        return _whisper_processor, _whisper_model, _device
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}")


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """Extract audio from video using moviepy."""
    try:
        # Try moviepy v2.x first
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            output_audio_path,
            fps=44100,
            nbytes=2,
            codec='pcm_s16le',
            logger=None
        )
        video.close()
        return True
    except Exception as e:
        raise RuntimeError(f"Error extracting audio from {video_path}: {e}")


def split_audio_on_silence(
    audio_path: str,
    min_dur: float,
    max_dur: float,
    mode: str,
    min_silence_duration: float = 0.5,
    padding_duration: float = 0.2,
    silence_threshold: int = 45
) -> List[np.ndarray]:
    """
    Split audio into chunks based on silence detection.
    
    Args:
        audio_path: Path to audio file
        min_dur: Minimum chunk duration in seconds
        max_dur: Maximum chunk duration in seconds
        mode: 'TTS' or 'STT'
        min_silence_duration: Minimum pause to consider as split point
        padding_duration: Silence added at start/end of segments
        silence_threshold: dB threshold for silence detection
    
    Returns:
        List of audio chunks as numpy arrays
    """
    MIN_SILENCE_DURATION = min_silence_duration
    PAD_DURATION = padding_duration
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Detect non-silent intervals
    top_db = silence_threshold
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # Merge intervals that are too close
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
    
    # Build chunks
    chunks = []
    current_chunk_parts = []
    current_len = 0
    
    min_samples = int(min_dur * sr)
    max_samples = int(max_dur * sr)
    pad_samples = int(PAD_DURATION * sr)
    join_pause_samples = int(0.1 * sr)
    join_pause = np.zeros(join_pause_samples)
    
    for start, end in merged_intervals:
        segment = y[start:end]
        seg_len = len(segment)
        
        # Skip segments that are too long
        if seg_len > max_samples:
            continue
        
        added_len = seg_len
        if current_chunk_parts:
            added_len += join_pause_samples
        
        if current_len + added_len <= max_samples:
            if current_chunk_parts:
                current_chunk_parts.append(join_pause)
                current_len += join_pause_samples
            
            current_chunk_parts.append(segment)
            current_len += seg_len
        else:
            if current_len >= min_samples:
                full_audio = np.concatenate(current_chunk_parts)
                full_audio = np.pad(full_audio, (pad_samples, pad_samples), mode='constant')
                chunks.append(full_audio)
            
            current_chunk_parts = [segment]
            current_len = seg_len
    
    # Last chunk
    if current_chunk_parts and current_len >= min_samples:
        full_audio = np.concatenate(current_chunk_parts)
        full_audio = np.pad(full_audio, (pad_samples, pad_samples), mode='constant')
        chunks.append(full_audio)
    
    return chunks


def transcribe_audio(audio_data: np.ndarray, sample_rate: int = 44100) -> str:
    """Transcribe audio using Whisper."""
    processor, model, device = _load_whisper_model()
    
    import torch
    
    # Resample to 16kHz for Whisper
    chunk_16k = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    
    # Process with Whisper
    input_features = processor(
        chunk_16k,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription
    predicted_ids = model.generate(
        input_features,
        language="ro",
        task="transcribe"
    )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Clean text
    text = transcription.strip().replace('\n', ' ').replace('  ', ' ')
    
    return text


def process_videos_to_dataset(
    video_paths: List[str],
    project_folder: str,
    mode: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    whisper_model: str = None,
    min_dur: int = None,
    max_dur: int = None,
    min_silence_duration: float = 0.5,
    padding_duration: float = 0.2,
    silence_threshold: int = 45
) -> Tuple[int, str]:
    """
    Process video files and create dataset.
    
    Args:
        video_paths: List of paths to video files
        project_folder: Destination folder for the project
        mode: 'TTS' or 'STT'
        progress_callback: Optional callback for progress updates
        whisper_model: HuggingFace model name
        min_dur: Minimum segment duration
        max_dur: Maximum segment duration
        min_silence_duration: Minimum pause to consider as split point
        padding_duration: Silence added at start/end
        silence_threshold: dB threshold for silence detection
    
    Returns:
        Tuple of (valid_count, csv_path)
    """
    os.makedirs(project_folder, exist_ok=True)
    
    # Set duration limits based on mode if not provided
    if min_dur is None:
        min_dur = 3
    if max_dur is None:
        max_dur = 10 if mode == 'TTS' else 30
    
    csv_path = os.path.join(project_folder, 'metadata.csv')
    
    # Find the next available index by checking existing files and CSV
    next_index = 0
    
    # Check existing wav files in folder
    import glob
    existing_wavs = glob.glob(os.path.join(project_folder, "*.wav"))
    for wav_file in existing_wavs:
        filename = os.path.basename(wav_file)
        try:
            idx = int(filename.replace('.wav', ''))
            next_index = max(next_index, idx + 1)
        except ValueError:
            pass
    
    # Also check CSV for highest index
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    wav_name = line.split('|')[0].strip()
                    try:
                        idx = int(wav_name.replace('.wav', ''))
                        next_index = max(next_index, idx + 1)
                    except ValueError:
                        pass
    
    valid_count = next_index
    entries_added = 0  # Track how many new entries we add
    
    total_videos = len(video_paths)
    
    with open(csv_path, 'a', encoding='utf-8', newline='') as csv_file:
        for v_idx, video_path in enumerate(video_paths):
            video_name = Path(video_path).name
            
            if progress_callback:
                progress_callback(
                    int((v_idx / total_videos) * 100),
                    f"Processing video: {video_name}"
                )
            
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_audio_path = tmp.name
            
            try:
                # Extract audio
                if progress_callback:
                    progress_callback(
                        int((v_idx / total_videos) * 100),
                        f"Extracting audio from {video_name}..."
                    )
                
                extract_audio_from_video(video_path, temp_audio_path)
                
                # Split audio
                if progress_callback:
                    progress_callback(
                        int((v_idx / total_videos) * 100),
                        f"Splitting audio from {video_name}..."
                    )
                
                chunks = split_audio_on_silence(
                    temp_audio_path, min_dur, max_dur, mode,
                    min_silence_duration, padding_duration, silence_threshold
                )
                
                # Transcribe each chunk
                for c_idx, chunk_data in enumerate(chunks):
                    if progress_callback:
                        progress_callback(
                            int(((v_idx + c_idx / len(chunks)) / total_videos) * 100),
                            f"Transcribing chunk {c_idx + 1}/{len(chunks)} from {video_name}..."
                        )
                    
                    # Save audio chunk
                    wav_filename = f"{valid_count:012d}.wav"
                    wav_path = os.path.join(project_folder, wav_filename)
                    sf.write(wav_path, chunk_data, 44100, subtype='PCM_16')
                    
                    # Transcribe
                    try:
                        text = transcribe_audio(chunk_data)
                        
                        if text and len(text) >= 2:
                            # Write to CSV
                            csv_file.write(f"{wav_filename}|{text}|{text}\n")
                            csv_file.flush()
                            valid_count += 1
                            entries_added += 1
                        else:
                            # Empty transcription, delete file
                            os.remove(wav_path)
                    except Exception as e:
                        # Transcription failed, delete file
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(
                        int((v_idx / total_videos) * 100),
                        f"Error processing {video_name}: {e}"
                    )
            finally:
                # Cleanup temp audio
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
    
    if progress_callback:
        progress_callback(100, f"Done! Added {entries_added} new samples (total: {valid_count}).")
    
    return entries_added, csv_path

