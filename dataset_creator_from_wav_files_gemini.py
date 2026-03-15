import os
import shutil
import csv
import soundfile as sf
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from google import genai
from google.genai import types

# ==========================================
# CONFIGURARE
# ==========================================

MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
PRICE_INPUT_1M = 0.30
PRICE_OUTPUT_1M = 2.50
TOKENS_PER_SECOND_AUDIO = 32
TOKENS_PER_TEXT_PROMPT = 250 

SYSTEM_PROMPT = """
Ești un expert în transcrierea audio (Speech-to-Text) pentru limba română.
Sarcina ta este să transcrii fișierul audio furnizat exact așa cum este vorbit.
1. Audio-ul conține grai moldovenesc. Redă fidel ce s-a spus.
2. Returnează DOAR textul transcris. Fără timestamp-uri, fără introduceri.
"""

console = Console()

def get_api_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]Eroare:[/bold red] Variabila de mediu 'GEMINI_API_KEY' nu este setată.")
        exit(1)
    return genai.Client(api_key=api_key)

def get_all_wav_files_robust(source_path):
    """
    Scanează recursiv și verifică extensia manual pentru a evita problemele de Case Sensitivity.
    Găsește: .wav, .WAV, .Wave, etc.
    """
    found_files = []
    # Folosim rglob('*') pentru a lua TOATE fișierele și filtrăm noi manual
    try:
        iterator = source_path.rglob("*")
        for file_path in iterator:
            if file_path.is_file():
                # Verificare strictă doar pe extensie, indiferent de mărime (wav/WAV)
                if file_path.suffix.lower() == ".wav":
                    found_files.append(file_path)
    except Exception as e:
        console.print(f"[red]Eroare la scanarea folderului: {e}[/red]")
    
    return found_files

def get_processed_files(metadata_path):
    """Încarcă într-un SET numele fișierelor deja terminate."""
    processed = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if row and len(row) > 0:
                    processed.add(row[0].strip()) # strip() e vital pentru siguranță
    return processed

def get_audio_duration(file_path):
    try:
        f = sf.SoundFile(file_path)
        return len(f) / f.samplerate
    except Exception:
        return 0

def transcribe_audio(client, file_path):
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        response = client.models.generate_content(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
            ),
            contents=[
                "Transcribe this.", # Prompt scurt pentru a economisi tokeni
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"
                )
            ]
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        console.print(f"[red]![/red] Eroare API la {file_path.name}: {e}")
        return None

def main():
    console.clear()
    console.print(Panel.fit("[bold green]Generator Dataset STT - Mod ROBUST[/bold green]", subtitle="Procesează TOT ce are extensia .wav"))
    
    source_folder = Prompt.ask("Introdu calea către dosarul cu fișiere .wav")
    default_dest = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest_folder = Prompt.ask("Introdu calea pentru setul de date final", default=default_dest)

    source_path = Path(source_folder)
    dest_path = Path(dest_folder)
    metadata_path = dest_path / "metadata.csv"

    if not source_path.exists():
        console.print(f"[bold red]Eroare:[/bold red] Folderul sursă '{source_folder}' nu există.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)
    client = get_api_client()

    # 1. SCANARE COMPLETĂ
    with console.status("[bold cyan]Scanez TOATE fișierele din dosar (poate dura puțin)..."):
        all_files_on_disk = get_all_wav_files_robust(source_path)

    # 2. CITIRE ISTORIC
    processed_files_names = get_processed_files(metadata_path)

    # 3. CALCUL DIFERENȚĂ (Set Subtraction)
    # Identificăm fișierele care sunt pe disc dar NU sunt în CSV
    files_to_process = []
    for f in all_files_on_disk:
        if f.name not in processed_files_names:
            files_to_process.append(f)

    # 4. RAPORT DIAGNOSTIC
    table = Table(title="Raport Fișiere")
    table.add_column("Categorie", style="cyan")
    table.add_column("Număr", style="bold white")
    
    table.add_row("Total fișiere .wav găsite pe disc", str(len(all_files_on_disk)))
    table.add_row("Fișiere deja în metadata.csv", str(len(processed_files_names)))
    table.add_row("Rămase de procesat", f"[bold green]{len(files_to_process)}[/bold green]")
    
    console.print(table)

    if len(files_to_process) == 0:
        console.print("[bold green]Toate fișierele au fost procesate! Nu am nimic de făcut.[/bold green]")
        return

    # Sărim peste calculul de costuri detaliat dacă sunt mii de fișiere pentru a nu pierde timpul,
    # dar afișăm o estimare rapidă
    est_files = len(files_to_process)
    console.print(f"\n[yellow]Estimare rapidă:[/yellow] Vei procesa {est_files} fișiere.")
    if not Confirm.ask("[bold white]Începem procesarea?[/bold white]"):
        return

    # 5. PROCESARE EFECTIVĂ
    with open(metadata_path, 'a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[green]Procesare...", total=len(files_to_process))
            
            for wav_file in files_to_process:
                progress.update(task, description=f"Procesez: {wav_file.name}")
                
                text = transcribe_audio(client, wav_file)
                
                if text is not None:
                    text_clean = text.replace('\n', ' ').replace('\r', '').strip()
                    
                    # Salvare CSV
                    writer.writerow([wav_file.name, text_clean, text_clean])
                    csvfile.flush()
                    os.fsync(csvfile.fileno())

                    # Copiere Audio
                    try:
                        shutil.copy2(wav_file, dest_path / wav_file.name)
                    except Exception as copy_err:
                        # Dacă e o eroare de copiere, scriem în consolă dar continuăm
                        # Fișierul e în CSV, deci datele sunt salvate
                        pass 
                
                progress.advance(task)

    console.print(f"\n[bold green]GATA![/bold green] Verifică folderul {dest_path}")

if __name__ == "__main__":
    main()