import fitz  # PyMuPDF
import os
import sys
import csv
import glob
import time
import io
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
import vision_config as config

# Încercăm să importăm ollama
try:
    import ollama
except ImportError:
    print("Eroare: Librăria 'ollama' nu este instalată.")
    print("Te rog rulează: pip install ollama")
    sys.exit(1)

console = Console()

def check_ollama_connection():
    """Verifică dacă serverul Ollama rulează și dacă modelul există."""
    try:
        models = ollama.list()
        model_names = [m['name'] for m in models['models']]
        
        # Verificare simplificată a numelui (ex: llava:latest vs llava)
        found = any(config.MODEL_NAME in name for name in model_names)
        
        if not found:
            console.print(f"[yellow]Avertisment: Modelul '{config.MODEL_NAME}' nu pare să fie descărcat.[/yellow]")
            console.print(f"Te rog rulează în terminal: [bold]ollama pull {config.MODEL_NAME}[/bold]")
            return False
        return True
    except Exception as e:
        console.print(f"[red]Eroare conectare Ollama: {e}[/red]")
        console.print("Asigură-te că aplicația Ollama rulează.")
        return False

def pdf_to_images(pdf_path, temp_dir="temp_pages"):
    """Convertește paginile PDF în imagini PNG temporare."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300) # DPI mare pentru claritate
        image_path = os.path.join(temp_dir, f"page_{page_num:04d}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        
    doc.close()
    return image_paths

def process_image_with_ollama(image_path, mode):
    """Trimite imaginea la Ollama și primește CSV-ul."""
    
    system_prompt = config.SYSTEM_PROMPT_TTS if mode == "TTS" else config.SYSTEM_PROMPT_STT
    
    # Instrucțiune specifică pentru imaginea curentă
    user_message = "Analizează această imagine și extrage textul conform regulilor din System Prompt. Returnează doar liniile CSV."

    try:
        response = ollama.chat(
            model=config.MODEL_NAME,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_message,
                    'images': [image_path]
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        console.print(f"[red]Eroare procesare AI pentru {image_path}: {e}[/red]")
        return ""

def parse_ai_response(response_text, filename_prefix):
    """Curăță răspunsul AI și extrage liniile valide CSV."""
    lines = response_text.strip().split('\n')
    valid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Ignorăm liniile care nu conțin separatorul pipe sau sunt comentarii
        if '|' not in line:
            continue
            
        parts = line.split('|')
        if len(parts) < 3:
            # Uneori modelul poate uita numele fișierului, încercăm să reparăm
            if len(parts) == 2:
                # Presupunem TextBrut|TextNormalizat
                valid_rows.append([filename_prefix, parts[0].strip(), parts[1].strip()])
            continue
            
        # Format corect: Fisier|TextBrut|TextNormalizat
        # Luăm ultimele 2 părți ca text, prima ca ID (dar o suprascriem cu ID-ul nostru unic dacă e nevoie)
        # De fapt, modelul returnează img_name|text|text. Noi vrem să salvăm curat.
        
        raw_text = parts[1].strip()
        norm_text = parts[2].strip()
        
        if len(raw_text) > 5: # Filtru minim de lungime
            valid_rows.append([filename_prefix, raw_text, norm_text])
            
    return valid_rows

def main():
    console.print(Panel.fit("[bold green]AI Vision Dataset Builder[/bold green]\nExtragere și Normalizare simultană folosind Ollama"))

    if not check_ollama_connection():
        return

    # 1. Selectare PDF
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        console.print("[red]Nu am găsit niciun fișier PDF în folderul curent![/red]")
        return

    console.print("Fișiere PDF disponibile:")
    for idx, f in enumerate(pdf_files):
        console.print(f"{idx + 1}. {f}")
    
    pdf_idx = IntPrompt.ask("Alege numărul fișierului PDF", choices=[str(i+1) for i in range(len(pdf_files))])
    selected_pdf = pdf_files[pdf_idx - 1]

    # 2. Selectare Mod
    console.print("\nSelectează modul de lucru:")
    console.print("1. [cyan]TTS (Text-to-Speech)[/cyan] - Normalizare strictă (matematică expandată, fără paranteze)")
    console.print("2. [magenta]STT (Speech-to-Text)[/magenta] - Normalizare relaxată (păstrează cifre, ghilimele românești)")
    
    mode_choice = IntPrompt.ask("Alege modul", choices=["1", "2"])
    mode = "TTS" if mode_choice == 1 else "STT"
    
    output_csv = "metadata_vision_tts.csv" if mode == "TTS" else "metadata_vision_stt.csv"

    # 3. Procesare
    temp_dir = "temp_vision_processing"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # Pasul 1: Conversie PDF -> Imagini
        task_convert = progress.add_task(f"[green]Conversie PDF: {selected_pdf}...", total=None)
        image_paths = pdf_to_images(selected_pdf, temp_dir)
        progress.update(task_convert, total=len(image_paths), completed=len(image_paths))
        
        # Pasul 2: Procesare AI
        task_ai = progress.add_task(f"[blue]Procesare AI ({mode})...", total=len(image_paths))
        
        all_data = []
        
        for i, img_path in enumerate(image_paths):
            # Nume unic pentru fișierul audio care va fi generat ulterior
            file_id = f"{os.path.splitext(selected_pdf)[0]}_{i+1:04d}"
            
            response_text = process_image_with_ollama(img_path, mode)
            rows = parse_ai_response(response_text, file_id)
            
            # Adăugăm index la ID pentru a fi unici per propoziție
            for j, row in enumerate(rows):
                unique_id = f"{row[0]}_{j+1}"
                all_data.append([unique_id, row[1], row[2]])
            
            progress.advance(task_ai)
            
            # Curățenie imediată imagine procesată (opțional, pentru spațiu)
            try:
                os.remove(img_path)
            except:
                pass

    # 4. Salvare CSV
    if all_data:
        file_exists = os.path.exists(output_csv)
        with open(output_csv, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            if not file_exists:
                writer.writerow(['ID', 'Text Brut', 'Text Normalizat'])
            writer.writerows(all_data)
        
        console.print(f"\n[bold green]Succes![/bold green] Au fost extrase {len(all_data)} segmente.")
        console.print(f"Datele au fost salvate în: [yellow]{output_csv}[/yellow]")
    else:
        console.print("\n[red]Nu s-au extras date valide. Verifică dacă modelul AI funcționează corect.[/red]")

    # 5. Curățenie finală folder temp
    try:
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir) # Șterge doar dacă e gol, dar noi am șters imaginile pe parcurs
    except:
        pass

if __name__ == "__main__":
    main()
