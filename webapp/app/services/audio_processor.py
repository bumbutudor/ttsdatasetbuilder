"""Audio processing utilities."""
import numpy as np


def trim_silence(wav_path: str, mode: str = "TTS"):
    """
    Trim silence from audio file.
    
    Args:
        wav_path: Path to WAV file
        mode: 'TTS' for strict trimming, 'STT' for relaxed
    """
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(wav_path, sr=None)
        
        # Set parameters based on mode
        if mode == 'TTS':
            top_db = 30  # Strict
            pad_duration = 0.1
        else:
            top_db = 20  # Relaxed
            pad_duration = 0.5
        
        # Trim silence
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        
        # Add padding
        pad_len = int(sr * pad_duration)
        yt = np.pad(yt, (pad_len, pad_len), mode='constant')
        
        # Save
        sf.write(wav_path, yt, sr)
        
        return True
    except Exception as e:
        print(f"Error trimming audio: {e}")
        return False


def get_audio_duration(wav_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=None)
        return len(y) / sr
    except Exception:
        return 0.0

