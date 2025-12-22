#!/usr/bin/env python3
"""Calculează durata totală (ore) pentru fișiere .wav dintr-un director.

Utilizare:
    python total_wav_hours.py /cale/catre/folder --verbose

"""
import os
import wave
import contextlib
import argparse


def get_wav_duration(path):
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return None


def format_hms(seconds: float):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"


def main():
    p = argparse.ArgumentParser(description="Sum WAV durations in a folder and print total hours")
    p.add_argument('path', nargs='?', default='.', help='Folder to scan for .wav files')
    p.add_argument('--verbose', '-v', action='store_true', help='Show per-file durations')
    args = p.parse_args()

    root = os.path.abspath(args.path)
    total_seconds = 0.0
    count = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith('.wav'):
                continue
            fpath = os.path.join(dirpath, fname)
            dur = get_wav_duration(fpath)
            if dur is None:
                skipped += 1
                continue
            total_seconds += dur
            count += 1
            if args.verbose:
                print(f"{fpath}: {dur:.2f}s")

    hours = total_seconds / 3600.0
    print(f"Found {count} .wav files (skipped: {skipped}).")
    print(f"Total seconds: {total_seconds:.2f}")
    print(f"Total hours: {hours:.4f} h")
    print(f"Formatted: {format_hms(total_seconds)}")


if __name__ == '__main__':
    main()
