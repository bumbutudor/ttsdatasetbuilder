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
from rich.prompt import IntPrompt
import vision_config as config

# Încercăm să importăm ollama
try:
    import ollama
except ImportError:
    print("Eroare: Librăria 'ollama' nu este instalată.")
    print("Te rog rulează: pip install ollama")
    sys.exit(1)

console = Console()


def _extract_model_names(models_response):
    """Returnează o listă de nume de modele din ollama.list(), compatibil cu mai multe versiuni."""
    # Versiuni mai noi: ListResponse(models=[Model(model='llava:latest', ...), ...])
    if hasattr(models_response, "models"):
        names = []
        for m in getattr(models_response, "models") or []:
            if hasattr(m, "model"):
                names.append(getattr(m, "model"))
            elif hasattr(m, "name"):
                names.append(getattr(m, "name"))
            elif isinstance(m, dict):
                names.append(m.get("model") or m.get("name"))
        return [n for n in names if n]

    # Versiuni mai vechi: dict {'models': [{'name': 'llava:latest', ...}, ...]}
    if isinstance(models_response, dict):
        models = models_response.get("models") or []
        out = []
        for m in models:
            if isinstance(m, dict):
                out.append(m.get("model") or m.get("name"))
            else:
                out.append(getattr(m, "model", None) or getattr(m, "name", None))
        return [n for n in out if n]

    return []

def check_ollama_connection():
    """Verifică dacă serverul Ollama rulează și dacă modelul există."""
    try:
        models = ollama.list()
        model_names = _extract_model_names(models)

        # Verificare simplificată a numelui (ex: llava:latest vs llava)
        wanted = (config.MODEL_NAME or "").strip()
        found = any(
            name == wanted or name.startswith(wanted + ":") or (wanted and wanted in name)
            for name in model_names
        )
        
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
    user_message = (
        "Analizează această imagine și extrage textul conform regulilor din System Prompt. "
        "Returnează doar liniile CSV în format: TextBrut|TextNormalizat (2 coloane)."
    )

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


def process_text_with_ollama(text_chunk, mode):
    """Trimite text (nu imagine) la Ollama și primește CSV-ul."""
    system_prompt = config.SYSTEM_PROMPT_TTS if mode == "TTS" else config.SYSTEM_PROMPT_STT

    user_message = (
        "Împarte textul următor în segmente potrivite conform regulilor din System Prompt și "
        "returnează doar liniile CSV în format: TextBrut|TextNormalizat (2 coloane).\n\nTEXT:\n"
        + text_chunk
    )

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
                    'content': user_message
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        console.print(f"[red]Eroare procesare AI (text): {e}[/red]")
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
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 2:
            continue

        # Acceptăm atât format cu 2 coloane (Brut|Normalizat) cât și 3+ (Fisier|Brut|Normalizat)
        if len(parts) == 2:
            raw_text, norm_text = parts[0], parts[1]
        else:
            raw_text, norm_text = parts[-2], parts[-1]
        
        if len(raw_text) > 5: # Filtru minim de lungime
            valid_rows.append([filename_prefix, raw_text, norm_text])
            
    return valid_rows

def main():
    console.print(Panel.fit("[bold green]AI Vision Dataset Builder[/bold green]\nExtragere și Normalizare simultană folosind Ollama"))

    if not check_ollama_connection():
        return

    # UX identic cu main_generate_csv.py
    console.print("\nSelect dataset type:")
    console.print("1. [cyan]TTS[/cyan] (Text-to-Speech) - Normalizare strictă")
    console.print("2. [green]STT[/green] (Speech-to-Text) - Normalizare relaxată")
    mode_input = input("Choice (1/2): ").strip()

    if mode_input == '2':
        mode = 'STT'
        folder_prefix = 'project_VISION_STT_'
    else:
        mode = 'TTS'
        folder_prefix = 'project_VISION_TTS_'

    console.print(f"Selected mode: [bold]{mode}[/bold]")

    app_folder = os.path.dirname(os.path.realpath(__file__))

    # Folder proiect (output)
    from datetime import datetime
    now = datetime.now()
    default_project_folder = os.path.join(app_folder, folder_prefix + now.strftime("%d%m%Y_%H%M%S"))
    console.print("Please select a [red]project folder[/red] (default [i]%s[/i])." % default_project_folder)
    in_project_folder = input()
    if not in_project_folder:
        in_project_folder = default_project_folder
    project_folder = in_project_folder
    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    console.print("Project folder is %s" % project_folder)

    # Tipuri fișiere de procesat
    console.print("Please select the file types you want to load (t=text, p=pdf, tp=both) (default [i]tp[/i])")
    in_file_types = input().strip()
    if not in_file_types:
        in_file_types = "tp"
    if 't' not in in_file_types and 'p' not in in_file_types:
        in_file_types = "tp"

    texts_folder = os.path.join(app_folder, "texts")
    if not os.path.exists(texts_folder):
        console.print("[red]Folderul 'texts' nu există lângă script.[/red]")
        return

    txt_files = []
    pdf_files = []
    if 't' in in_file_types:
        txt_files = glob.glob(os.path.join(texts_folder, "*.txt"))
        console.print("Found %d text files" % (len(txt_files)))
    if 'p' in in_file_types:
        pdf_files = glob.glob(os.path.join(texts_folder, "*.pdf"))
        console.print("Found %d pdf files" % (len(pdf_files)))

    if not txt_files and not pdf_files:
        console.print("[red]Nu am găsit fișiere de procesat în 'texts'.[/red]")
        return

    # Output CSV
    output_csv = os.path.join(project_folder, 'metadata.csv')

    # Counter global pentru ID-uri tip .wav (ca în main_generate_csv.py)
    valid_count = 0

    # Temp folder (în proiect) - va fi șters la final
    temp_dir = os.path.join(project_folder, "temp_vision_processing")

    all_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        # 1) TXT -> LLM (fără imagini)
        if txt_files:
            task_txt = progress.add_task(f"[blue]Procesare TXT cu AI ({mode})...", total=len(txt_files))
            for tf in txt_files:
                try:
                    with open(tf, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    console.print(f"[red]Eroare citire TXT {tf}: {e}[/red]")
                    progress.advance(task_txt)
                    continue

                # Chunking simplu ca să evităm prompt-uri uriașe
                chunk_size = 6000
                for start in range(0, len(content), chunk_size):
                    chunk = content[start:start+chunk_size]
                    if not chunk.strip():
                        continue
                    resp = process_text_with_ollama(chunk, mode)
                    rows = parse_ai_response(resp, os.path.basename(tf))
                    for row in rows:
                        wav_file_name = (str(valid_count) + '.wav').rjust(12, '0')
                        all_data.append([wav_file_name, row[1], row[2]])
                        valid_count += 1

                progress.advance(task_txt)

        # 2) PDF -> imagini -> Vision LLM
        if pdf_files:
            task_pdf = progress.add_task(f"[green]Procesare PDF cu Vision AI ({mode})...", total=len(pdf_files))
            for pdf_path in pdf_files:
                # Conversie PDF -> Imagini
                try:
                    image_paths = pdf_to_images(pdf_path, temp_dir)
                except Exception as e:
                    console.print(f"[red]Eroare conversie PDF {pdf_path}: {e}[/red]")
                    progress.advance(task_pdf)
                    continue

                for img_path in image_paths:
                    response_text = process_image_with_ollama(img_path, mode)
                    rows = parse_ai_response(response_text, os.path.basename(img_path))

                    for row in rows:
                        wav_file_name = (str(valid_count) + '.wav').rjust(12, '0')
                        all_data.append([wav_file_name, row[1], row[2]])
                        valid_count += 1

                    # Curățenie imediată imagine procesată
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass

                progress.advance(task_pdf)

    # Salvare CSV
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

    # Curățenie finală folder temp
    try:
        if os.path.exists(temp_dir):
            # ar trebui să fie gol deoarece ștergem PNG-urile pe parcurs
            os.rmdir(temp_dir)
    except Exception:
        pass

    return

if __name__ == "__main__":
    main()
