#!/usr/bin/env python3
"""
Script simplu care convertează fișiere WAV de 44100 Hz în 16000 Hz.

Folosește `ffmpeg` pentru resampling și suprascrie fișierele originale
în loc (in-place). `ffmpeg` trebuie să fie instalat și în PATH.
"""
import sys
import os
from pathlib import Path
import subprocess
import shutil
from typing import Optional

TARGET_SR = 16000
EXPECTED_SR = 44100


def _get_sample_rate_via_ffprobe(path: Path) -> Optional[int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        s = out.decode().strip()
        if not s:
            return None
        return int(s.splitlines()[0])
    except Exception:
        return None


def convert_file(in_path: Path, out_path: Path, target_sr: int = TARGET_SR) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print(f"ffmpeg nu este disponibil în PATH. Nu pot converti {in_path.name}.")
        return

    # check sample rate and skip if not the expected one
    sr = _get_sample_rate_via_ffprobe(in_path)
    if sr is None:
        print(f"Nu am putut determina sample rate pentru {in_path.name}; sar peste.")
        return
    if sr != EXPECTED_SR:
        print(f"Skipping {in_path.name}: sample rate {sr} Hz (expected {EXPECTED_SR} Hz)")
        return

    # write to a temp file in same directory then replace original to be safe
    tmp_out = in_path.with_suffix(in_path.suffix + ".tmp.wav")
    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path), "-ar", str(target_sr), "-acodec", "pcm_s16le", str(tmp_out)]
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0 and tmp_out.exists():
            # replace original file
            try:
                os.replace(str(tmp_out), str(out_path))
                print(f"Converted (ffmpeg): {in_path.name}")
            except Exception as e:
                print(f"Eroare la înlocuirea fișierului {in_path.name}: {e}")
                # cleanup tmp
                try:
                    tmp_out.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            print(f"ffmpeg error for {in_path.name}: returncode={res.returncode}")
            try:
                tmp_out.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as e:
        print(f"Eroare la {in_path.name} (ffmpeg): {e}")


def main():
    inp = input("Introduceți calea către directorul cu fișiere WAV (44.1 kHz): ").strip()
    if not inp:
        print("Cale goală. Ieșire.")
        sys.exit(1)
    p = Path(inp).expanduser()
    if not p.exists() or not p.is_dir():
        print("Cale invalidă sau nu este un director.")
        sys.exit(1)
    wav_files = list(p.rglob("*.wav"))
    if not wav_files:
        print("Nu s-au găsit fișiere .wav în directorul specificat.")
        sys.exit(0)

    print(f"Găsite {len(wav_files)} fișiere WAV. Încep conversia in-place...")
    converted = 0
    for f in wav_files:
        # convertim și suprascriem fișierul original
        convert_file(f, f, TARGET_SR)
        converted += 1
    print(f"Conversie finalizată. Fișiere procesate: {converted}")


if __name__ == "__main__":
    main()
