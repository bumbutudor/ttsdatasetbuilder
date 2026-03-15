"""Text normalization services for TTS and STT."""
import sys
import os
from typing import Literal

# În Docker, python path este setat corect la root-ul aplicației (/app).
# Fișierele sunt în pachetul app.services, deci le importăm direct de acolo.

try:
    # Încercăm importul absolut (pentru rularea din webapp/Docker)
    from app.services.main_normalize_text_TTS import normalize_tts
except ImportError:
    try:
        # Fallback: Încercăm importul relativ (dacă rulăm fișierul local sau diferit)
        from .main_normalize_text_TTS import normalize_tts
    except ImportError as e:
        print(f"Warning: Could not import normalize_tts: {e}")
        # Funcție dummy dacă importul eșuează
        def normalize_tts(text: str) -> str:
            return text

try:
    # Încercăm importul absolut (pentru rularea din webapp/Docker)
    from app.services.main_normalize_text_STT import normalize_stt
except ImportError:
    try:
        # Fallback: Încercăm importul relativ
        from .main_normalize_text_STT import normalize_stt
    except ImportError as e:
        print(f"Warning: Could not import normalize_stt: {e}")
        # Funcție dummy dacă importul eșuează
        def normalize_stt(text: str) -> str:
            return text


def normalize_text(text: str, mode: Literal["TTS", "STT"]) -> str:
    """Normalize text based on mode.
    
    Args:
        text: The text to normalize
        mode: Either "TTS" or "STT"
        
    Returns:
        Normalized text
    """
    if mode == "TTS":
        return normalize_tts(text)
    else:
        return normalize_stt(text)


# Aliases for compatibility
normalize_for_tts = normalize_tts
normalize_for_stt = normalize_stt