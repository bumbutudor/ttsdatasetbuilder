"""Text normalization services for TTS and STT.

This module imports directly from the original normalization scripts
in the project root. To modify normalization rules, edit:
- main_normalize_text_TTS.py (for TTS normalization)
- main_normalize_text_STT.py (for STT normalization)
"""
import sys
import os
from typing import Literal

# Add project root to path to import original scripts
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import normalization functions from original scripts
try:
    from main_normalize_text_TTS import normalize_tts
except ImportError as e:
    print(f"Warning: Could not import normalize_tts from main_normalize_text_TTS.py: {e}")
    def normalize_tts(text: str) -> str:
        return text

try:
    from main_normalize_text_STT import normalize_stt
except ImportError as e:
    print(f"Warning: Could not import normalize_stt from main_normalize_text_STT.py: {e}")
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
