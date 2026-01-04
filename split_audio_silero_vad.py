import os
import glob
import sys
import warnings
import numpy as np
import soundfile as sf
import librosa
from rich.console import Console
from rich.table import Table
from rich.progress import track
from datetime import datetime

# Verificare existență Torch (Obligatoriu pentru Silero)
try:
    import torch
    # Dezactivăm warning-urile specifice torch
    warnings.filterwarnings("ignore", category=UserWarning) 
except ImportError:
    print("Eroare: Acest script necesită 'torch' pentru Silero VAD.")
    print("Te rog instalează: pip install torch torchaudio")
    sys.exit(1)

console = Console()

def load_silero_model():
    """Încarcă modelul Silero VAD din torch hub."""
    try:
        console.print("[yellow]Se încarcă modelul Silero VAD...[/yellow]")
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False,
                                      trust_repo=True)
        return model, utils
    except Exception as e:
        console.print(f"[bold red]Nu s-a putut încărca Silero VAD: {e}[/bold red]")
        sys.exit(1)

def get_next_file_index(output_folder):
    """Calculează următorul index bazat pe fișierele existente."""
    existing_files = glob.glob(os.path.join(output_folder, "*.wav"))
    if not existing_files:
        return 0
    
    max_idx = -1
    for f in existing_files:
        try:
            base = os.path.basename(f).replace('.wav', '')
            if base.isdigit():
                idx = int(base)
                if idx > max_idx:
                    max_idx = idx
        except:
            continue
            
    return max_idx + 1

def process_file_with_silero(file_path, model, utils, min_dur, max_dur, target_sr):
    """
    Procesează un singur fișier audio folosind Librosa pentru încărcare (fix pentru eroarea torchaudio).
    """
    # Extragem utilitarele (ignorăm read_audio stricat)
    (get_speech_timestamps, _, _, _, _) = utils
    
    # --- PASUL 1: Încărcare pentru VAD (16000 Hz) ---
    try:
        # Folosim librosa pentru a citi fișierul la 16k
        wav_np, _ = librosa.load(file_path, sr=16000, mono=True)
        # Convertim în tensor Torch
        wav_16k = torch.from_numpy(wav_np)
    except Exception as e:
        console.print(f"[red]Eroare citire audio (16k) {os.path.basename(file_path)}: {e}[/red]")
        return []

    # --- PASUL 2: Detectare segmente ---
    try:
        # threshold=0.5 (sensibilitate standard)
        # min_speech_duration_ms=250 (ignorăm zgomotele foarte scurte)
        speech_timestamps = get_speech_timestamps(wav_16k, model, sampling_rate=16000, threshold=0.5, min_speech_duration_ms=250)
    except Exception as e:
         console.print(f"[red]Eroare execuție VAD pe {os.path.basename(file_path)}: {e}[/red]")
         return []
    
    if not speech_timestamps:
        return []

    # --- PASUL 3: Încărcare High Quality pentru tăiere ---
    try:
        wav_hq, _ = librosa.load(file_path, sr=target_sr)
    except Exception as e:
        console.print(f"[red]Eroare citire audio (HQ) {os.path.basename(file_path)}: {e}[/red]")
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

    current_chunk_start = adjusted_timestamps[0][0]
    current_chunk_end = adjusted_timestamps[0][1]
    
    # Buffer mic (0.1s) la capete pentru naturalețe
    context_samples = int(0.1 * target_sr)

    for i in range(1, len(adjusted_timestamps)):
        next_start, next_end = adjusted_timestamps[i]
        
        # Verificăm dacă adăugarea segmentului următor depășește MAX_DUR (30s)
        if (next_end - current_chunk_start) < max_samples:
            # Dacă nu depășește, unim segmentele
            current_chunk_end = next_end
        else:
            # Dacă depășește, salvăm ce am acumulat până acum
            s = max(0, current_chunk_start - context_samples)
            e = min(len(wav_hq), current_chunk_end + context_samples)
            segment = wav_hq[s:e]
            
            # Verificăm dacă segmentul rezultat respectă MIN_DUR (3s)
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

if __name__ == '__main__':
    table = Table(title="Audio Segmentation Tool")
    table.add_column("Setare", style="cyan")
    table.add_column("Valoare", style="green")
    table.add_row("Durata Min.", "3 secunde")
    table.add_row("Durata Max.", "30 secunde")
    table.add_row("Metodă", "Silero VAD (Fix Librosa I/O)")
    console.print(table)

    # 1. Folder Sursă
    console.print("\n[bold yellow]1. Folderul cu fișiere WAV sursă:[/bold yellow]")
    source_folder = input("Cale dosar intrare: ").strip().strip('"').strip("'")
    
    if not os.path.exists(source_folder):
        console.print(f"[bold red]Eroare:[/bold red] Dosarul '{source_folder}' nu există.")
        sys.exit(1)
        
    wav_files = glob.glob(os.path.join(source_folder, "*.wav"))
    if not wav_files:
        console.print("[red]Nu s-au găsit fișiere .wav în dosarul specificat.[/red]")
        sys.exit(1)

    # 2. Folder Destinație
    console.print("\n[bold yellow]2. Denumirea dosarului final:[/bold yellow]")
    dest_name = input("Nume dosar ieșire: ").strip()
    if not dest_name: dest_name = "dataset_seg_30s"
    
    app_path = os.path.dirname(os.path.realpath(__file__))
    dest_folder = os.path.join(app_path, dest_name)
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        console.print(f"[green]Creat dosar:[/green] {dest_folder}")
    else:
        console.print(f"[yellow]Dosar existent:[/yellow] {dest_folder} (voi continua numerotarea)")

    # 3. Setări Sample Rate
    console.print("\n[bold yellow]3. Sample Rate output:[/bold yellow]")
    console.print("   [1] 22050 Hz (TTS)")
    console.print("   [2] 44100 Hz (High Quality)")
    console.print("   [3] 16000 Hz (STT)")
    sr_choice = input("Alege (1/2/3) [default 1]: ").strip()
    
    if sr_choice == '2': target_sr = 44100
    elif sr_choice == '3': target_sr = 16000
    else: target_sr = 22050
    
    # --- CONFIGURARE DURATE (MODIFICAT AICI) ---
    min_dur = 3.0
    max_dur = 30.0 # Acum este setat la 30 secunde

    # Inițializare Model
    model, utils = load_silero_model()

    # Logica Resume
    log_file = os.path.join(dest_folder, "processed_sources.log")
    processed_sources = set()
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            processed_sources = set(line.strip() for line in f if line.strip())
            
    files_to_process = [f for f in wav_files if os.path.basename(f) not in processed_sources]
    global_index = get_next_file_index(dest_folder)
    
    console.print(f"\n[cyan]De procesat: {len(files_to_process)} fișiere. Start index: {global_index}[/cyan]\n")

    if not files_to_process:
        console.print("[green]Totul este deja procesat![/green]")
        sys.exit(0)

    # Bucla Principală
    with open(log_file, 'a', encoding='utf-8') as log_f:
        for file_path in track(files_to_process, description="Segmentare (3s-30s)..."):
            file_name = os.path.basename(file_path)
            
            chunks = process_file_with_silero(file_path, model, utils, min_dur, max_dur, target_sr)
            
            for chunk in chunks:
                out_filename = f"{str(global_index).zfill(6)}.wav"
                out_path = os.path.join(dest_folder, out_filename)
                
                sf.write(out_path, chunk, target_sr, subtype='PCM_16')
                global_index += 1
            
            log_f.write(file_name + "\n")
            log_f.flush()

    console.print(f"\n[bold green]Gata![/bold green] Segmentele (3s-30s) sunt în: {dest_folder}")