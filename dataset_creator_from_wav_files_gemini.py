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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from google import genai
from google.genai import types

# ==========================================
# CONFIGURARE
# ==========================================

MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# Prețuri per 1 milion tokeni
PRICE_INPUT_1M = 0.30
PRICE_OUTPUT_1M = 2.50

# Configurare tokeni audio (Conform documentației Gemini: 1 sec audio = 32 tokeni)
TOKENS_PER_SECOND_AUDIO = 32
# Estimare tokeni text prompt (System prompt + user prompt) per fișier
TOKENS_PER_TEXT_PROMPT = 250 

SYSTEM_PROMPT = """
Ești un expert în transcrierea audio (Speech-to-Text) pentru limba română.
Sarcina ta este să transcrii fișierul audio furnizat exact așa cum este vorbit.

Instrucțiuni specifice:
1. Audio-ul conține grai moldovenesc din Republica Moldova. Trebuie să recunoști și să transcrii corect cuvintele regionale sau accentul specific, păstrând acuratețea fonetică a limbii române standard acolo unde este posibil, dar redând fidel ce s-a spus.
2. Returnează DOAR textul transcris.
3. NU include timestamp-uri, etichete de vorbitor sau alte meta-informații.
4. NU folosi formatare Markdown (fără bold, italic, titluri).
5. NU adăuga introduceri sau concluzii (ex: "Iată transcrierea:").
6. Dacă există numere, scrie-le în format text (ex: "doi metri" în loc de "2 metri") doar dacă este ambiguu, altfel formatul standard e acceptat.
"""

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

console = Console()

def get_api_client():
    """Inițializează clientul Google GenAI."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]Eroare:[/bold red] Variabila de mediu 'GEMINI_API_KEY' nu este setată.")
        exit(1)
    return genai.Client(api_key=api_key)

def get_processed_files(metadata_path):
    """Citește fișierul metadata.csv pentru a vedea ce fișiere au fost deja procesate."""
    processed = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if row:
                    processed.add(row[0])
    return processed

def get_audio_duration(file_path):
    """Returnează durata fișierului audio în secunde folosind soundfile."""
    try:
        f = sf.SoundFile(file_path)
        return len(f) / f.samplerate
    except Exception:
        return 0

def calculate_costs(files):
    """Calculează tokenii și costurile estimate."""
    total_seconds = 0
    total_files = len(files)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("Calculez durata totală audio...", total=total_files)
        for file in files:
            total_seconds += get_audio_duration(file)
            progress.advance(task)

    # 1. Calcul Input Tokens
    # Audio tokens + Text prompt tokens
    audio_tokens = total_seconds * TOKENS_PER_SECOND_AUDIO
    text_tokens = total_files * TOKENS_PER_TEXT_PROMPT
    total_input_tokens = audio_tokens + text_tokens
    
    cost_input = (total_input_tokens / 1_000_000) * PRICE_INPUT_1M

    # 2. Calcul Output Tokens (Estimare)
    # Estimăm grosier: vorbire normală ~3-4 tokeni pe secundă de output text
    estimated_output_tokens = total_seconds * 4 
    cost_output = (estimated_output_tokens / 1_000_000) * PRICE_OUTPUT_1M

    total_cost = cost_input + cost_output

    return total_seconds, total_input_tokens, cost_input, estimated_output_tokens, cost_output, total_cost

def transcribe_audio(client, file_path):
    """Trimite audio către Gemini API și returnează textul."""
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
                "Transcribe this audio exactly.",
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"
                )
            ]
        )
        
        if response.text:
            return response.text.strip()
        else:
            return ""
            
    except Exception as e:
        console.print(f"\n[bold red]Eroare la procesarea {os.path.basename(file_path)}:[/bold red] {e}")
        return None

def main():
    console.print(Panel.fit("[bold green]Generator Dataset STT - Gemini Paid Plan[/bold green]", subtitle="Calcul automat costuri"))
    
    # 1. Configurare Căi
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

    wav_files = sorted(list(source_path.glob("*.wav")))
    if not wav_files:
        console.print("[yellow]Nu s-au găsit fișiere .wav în folderul specificat.[/yellow]")
        return

    # 2. Logică de Resume
    processed_files = get_processed_files(metadata_path)
    files_to_process = [f for f in wav_files if f.name not in processed_files]

    if not files_to_process:
        console.print("[green]Toate fișierele au fost deja procesate![/green]")
        return

    # 3. Calcul Costuri și Confirmare
    console.print(f"\n[bold cyan]Analizez {len(files_to_process)} fișiere pentru estimarea costurilor...[/bold cyan]")
    
    seconds, in_tok, in_cost, out_tok, out_cost, total_est = calculate_costs(files_to_process)

    # Afișare Tabel Costuri
    table = Table(title="Estimare Costuri Procesare (Gemini 2.5 Flash)")

    table.add_column("Metrică", style="cyan")
    table.add_column("Valoare", style="magenta")
    table.add_column("Cost Estimat ($)", style="green", justify="right")

    table.add_row(
        "Input (Audio + Prompt)", 
        f"{int(in_tok):,} tokeni\n({seconds/60:.2f} minute audio)", 
        f"${in_cost:.4f}"
    )
    table.add_row(
        "Output (Text generat)*", 
        f"~{int(out_tok):,} tokeni", 
        f"~${out_cost:.4f}"
    )
    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", "", f"[bold]~${total_est:.4f}[/bold]")

    console.print(table)
    console.print("[italic dim]*Estimarea output-ului este aproximativă, bazată pe densitatea medie a vorbirii.[/italic dim]\n")

    if not Confirm.ask("[bold yellow]Dorești să începi procesarea și să fii taxat?[/bold yellow]"):
        console.print("[red]Proces anulat de utilizator.[/red]")
        return

    # 4. Procesare Efectivă
    with open(metadata_path, 'a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[green]Procesare activă...", total=len(files_to_process))
            
            for wav_file in files_to_process:
                progress.update(task, description=f"Procesez: {wav_file.name}")
                
                text = transcribe_audio(client, wav_file)
                
                if text:
                    text_clean = text.replace('\n', ' ').strip()
                    writer.writerow([wav_file.name, text_clean, text_clean])
                    
                    csvfile.flush() 
                    os.fsync(csvfile.fileno())

                    shutil.copy2(wav_file, dest_path / wav_file.name)
                
                progress.advance(task)

    console.print(f"\n[bold green]Finalizat![/bold green] Setul de date este salvat în: {dest_path}")
    console.print(f"Metadata salvat în: {metadata_path}")

if __name__ == "__main__":
    main()