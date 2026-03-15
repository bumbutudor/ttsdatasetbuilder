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

# Încercăm să importăm librăriile necesare
try:
    import ollama
except ImportError:
    if config.AI_PROVIDER == "ollama":
        print("Eroare: Librăria 'ollama' nu este instalată.")
        print("Te rog rulează: pip install ollama")
        sys.exit(1)

try:
    from openai import OpenAI
    import base64
except ImportError:
    if config.AI_PROVIDER == "openai":
        print("Eroare: Librăria 'openai' nu este instalată.")
        print("Te rog rulează: pip install openai")
        sys.exit(1)

console = Console()


def _next_wav_index_from_metadata(csv_path: str) -> int:
    """Returnează următorul index numeric pentru numele wav din metadata.csv.

    Compatibil cu formatul folosit de main_generate_csv.py / main_generator.py:
    prima coloană este un nume de fișier de forma 000000000001.wav.
    """
    if not os.path.exists(csv_path):
        return 0

    max_idx = -1
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if not row:
                    continue
                wav_name = (row[0] or '').strip()
                if not wav_name.lower().endswith('.wav'):
                    continue
                # Extragem partea numerică dinainte de .wav
                num_part = wav_name[:-4]
                if not num_part.isdigit():
                    continue
                max_idx = max(max_idx, int(num_part))
    except Exception:
        # Dacă fișierul e corupt/alt format, nu blocăm execuția.
        return 0

    return max_idx + 1


def _load_image_bytes_for_ollama(image_path: str, max_dim: int = 1280) -> bytes:
    """Încarcă o imagine ca bytes pentru câmpul `images` din ollama.chat.

    - Dacă Pillow e disponibil, face resize + conversie JPEG (mai mic și mai stabil pentru modele).
    - Altfel, returnează bytes brute din fișier.
    """
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as im:
            im = im.convert('RGB')
            w, h = im.size
            scale = min(1.0, float(max_dim) / float(max(w, h)))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)))

            bio = io.BytesIO()
            im.save(bio, format='JPEG', quality=85, optimize=True)
            return bio.getvalue()
    except Exception:
        # Fallback: trimitem fișierul exact cum e.
        with open(image_path, 'rb') as f:
            return f.read()


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

def check_openai_connection():
    """Verifică dacă avem API Key pentru OpenAI."""
    if not config.OPENAI_API_KEY:
        console.print("[red]Eroare: OPENAI_API_KEY nu este setat în vision_config.py sau environment.[/red]")
        return False
    return True

def check_ai_connection():
    if config.AI_PROVIDER == "openai":
        return check_openai_connection()
    return check_ollama_connection()

def pdf_to_images(pdf_path, temp_dir="temp_pages"):
    """Convertește paginile PDF în imagini PNG temporare."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # DPI prea mare poate produce imagini uriașe și poate duce la erori 500 (OOM/crash) în model.
        # 200 este un compromis bun pentru text imprimat.
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(temp_dir, f"page_{page_num:04d}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        
    doc.close()
    return image_paths


def iter_pdf_page_images(pdf_path: str, temp_dir: str, dpi: int = 200):
    """Generează imagini pentru paginile PDF una câte una.

    Avantaj: nu randează toate paginile upfront, astfel începe procesarea AI imediat și
    bara de progres se mișcă după fiecare pagină procesată.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            image_path = os.path.join(temp_dir, f"page_{page_num:04d}.png")
            pix.save(image_path)
            yield page_num, image_path
    finally:
        doc.close()

def process_image_with_ollama(image_path, mode):
    """Trimite imaginea la Ollama și primește CSV-ul."""
    
    system_prompt = config.SYSTEM_PROMPT_TTS if mode == "TTS" else config.SYSTEM_PROMPT_STT
    
    # Instrucțiune specifică pentru imaginea curentă (pagină PDF)
    # Cerință: extrage TOT textul de pe pagină, dar o propoziție = o linie CSV.
    user_message = (
        "Analizează această imagine (o pagină) și extrage TOT textul relevant. "
        "Împarte în propoziții/segmente: o propoziție pe linie. "
        "Returnează DOAR liniile CSV în format: TextBrut|TextNormalizat (2 coloane). "
        "NU include titluri sau markdown. "
        "NU folosi caracterul | în interiorul textului (doar ca separator între cele 2 coloane)."
    )

    # IMPORTANT: API-ul Ollama pentru imagini așteaptă bytes (base64 în request), nu o cale de fișier.
    # În plus, imaginile foarte mari pot declanșa 500 (OOM/crash) în backend. Facem resize+JPEG când e posibil.
    try:
        image_bytes = _load_image_bytes_for_ollama(image_path, max_dim=1280)

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
                    'images': [image_bytes]
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        msg = str(e)
        # Retry mai agresiv pe cazuri tipice de 500.
        if '500' in msg or 'Internal Server Error' in msg:
            try:
                console.print(f"[yellow]500 la procesare; retry cu imagine mai mică pentru {os.path.basename(image_path)}...[/yellow]")
                image_bytes = _load_image_bytes_for_ollama(image_path, max_dim=896)
                response = ollama.chat(
                    model=config.MODEL_NAME,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message, 'images': [image_bytes]},
                    ]
                )
                return response['message']['content']
            except Exception as e2:
                console.print(f"[red]Eroare procesare AI pentru {image_path}: {e2}[/red]")
                return ""

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

def _encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_with_openai(image_path, mode):
    """Trimite imaginea la OpenAI și primește CSV-ul."""
    system_prompt = config.SYSTEM_PROMPT_TTS if mode == "TTS" else config.SYSTEM_PROMPT_STT
    
    user_message = (
        "Analizează această imagine (o pagină) și extrage TOT textul relevant. "
        "Împarte în propoziții/segmente: o propoziție pe linie. "
        "Returnează DOAR liniile CSV în format: TextBrut|TextNormalizat (2 coloane). "
        "NU include titluri sau markdown. "
        "NU folosi caracterul | în interiorul textului (doar ca separator între cele 2 coloane)."
    )

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        base64_image = _encode_image_to_base64(image_path)

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[red]Eroare procesare OpenAI pentru {image_path}: {e}[/red]")
        return ""

def process_text_with_openai(text_chunk, mode):
    """Trimite text la OpenAI și primește CSV-ul."""
    system_prompt = config.SYSTEM_PROMPT_TTS if mode == "TTS" else config.SYSTEM_PROMPT_STT

    user_message = (
        "Împarte textul următor în segmente potrivite conform regulilor din System Prompt și "
        "returnează doar liniile CSV în format: TextBrut|TextNormalizat (2 coloane).\n\nTEXT:\n"
        + text_chunk
    )

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[red]Eroare procesare OpenAI (text): {e}[/red]")
        return ""

def process_image(image_path, mode):
    if config.AI_PROVIDER == "openai":
        return process_image_with_openai(image_path, mode)
    return process_image_with_ollama(image_path, mode)

def process_text(text_chunk, mode):
    if config.AI_PROVIDER == "openai":
        return process_text_with_openai(text_chunk, mode)
    return process_text_with_ollama(text_chunk, mode)

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
    console.print(Panel.fit(f"[bold green]AI Vision Dataset Builder[/bold green]\nExtragere și Normalizare simultană folosind {config.AI_PROVIDER.upper()}"))

    if not check_ai_connection():
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

    # Output CSV (format compatibil main_generator.py: FARA HEADER)
    output_csv = os.path.join(project_folder, 'metadata.csv')

    # Counter global pentru ID-uri tip .wav (ca în main_generate_csv.py).
    # Dacă metadata.csv există deja, continuăm numerotarea ca să evităm coliziuni.
    valid_count = _next_wav_index_from_metadata(output_csv)

    # Temp folder (în proiect) - va fi șters la final
    temp_dir = os.path.join(project_folder, "temp_vision_processing")

    extracted_segments = 0

    # Estimăm workload-ul ca să nu pară că "stă" la 0% minute întregi.
    # - TXT: aproximăm nr. de chunk-uri din mărimea fișierului (bytes).
    # - PDF: numărăm paginile (len(doc)).
    chunk_size = 6000
    estimated_txt_chunks = 0
    for tf in txt_files:
        try:
            sz = os.path.getsize(tf)
            estimated_txt_chunks += max(1, int((sz + (chunk_size - 1)) // chunk_size))
        except Exception:
            estimated_txt_chunks += 1

    total_pdf_pages = 0
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            total_pdf_pages += len(doc)
            doc.close()
        except Exception:
            # Dacă nu putem deschide PDF-ul acum, lăsăm 0; va apărea eroarea la procesare.
            pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        # Deschidem o singură dată CSV-ul și scriem incremental (în timp real).
        # Fără header, compatibil cu main_generator.py.
        csv_file = open(output_csv, 'a', encoding='utf-8', newline='')
        writer = csv.writer(csv_file, delimiter='|')
        try:

            # 1) TXT -> LLM (fără imagini)
            if txt_files:
                task_txt = progress.add_task(
                    f"[blue]Procesare TXT cu AI ({mode})...",
                    total=max(1, estimated_txt_chunks),
                )
                for tf in txt_files:
                    try:
                        with open(tf, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        console.print(f"[red]Eroare citire TXT {tf}: {e}[/red]")
                        progress.advance(task_txt)
                        continue

                    # Chunking simplu ca să evităm prompt-uri uriașe
                    for start in range(0, len(content), chunk_size):
                        chunk = content[start:start+chunk_size]
                        if not chunk.strip():
                            progress.advance(task_txt)
                            continue

                        progress.update(
                            task_txt,
                            description=f"[blue]TXT ({mode})[/blue]: {os.path.basename(tf)} [{start//chunk_size + 1}]",
                        )
                        resp = process_text(chunk, mode)
                        rows = parse_ai_response(resp, os.path.basename(tf))

                        # Scriere incrementală (real-time)
                        if rows:
                            out_rows = []
                            for row in rows:
                                wav_file_name = f"{valid_count:012d}.wav"
                                out_rows.append([wav_file_name, row[1], row[2]])
                                valid_count += 1
                            writer.writerows(out_rows)
                            csv_file.flush()
                            extracted_segments += len(out_rows)

                        progress.advance(task_txt)

                # Reset descriere după TXT
                progress.update(task_txt, description=f"[blue]Procesare TXT cu AI ({mode})...[/blue]")

            # 2) PDF -> imagini -> Vision LLM
            if pdf_files:
                task_pdf = progress.add_task(
                    f"[green]Procesare PDF cu Vision AI ({mode})...",
                    total=max(1, total_pdf_pages),
                )
                for pdf_path in pdf_files:
                    # PDF -> imagine pagină-cu-pagină -> Vision LLM
                    try:
                        page_iter = iter_pdf_page_images(pdf_path, temp_dir, dpi=200)
                    except Exception as e:
                        console.print(f"[red]Eroare deschidere PDF {pdf_path}: {e}[/red]")
                        continue

                    for page_num, img_path in page_iter:
                        progress.update(
                            task_pdf,
                            description=f"[green]PDF ({mode})[/green]: {os.path.basename(pdf_path)} | page {page_num + 1}",
                        )
                        response_text = process_image(img_path, mode)
                        rows = parse_ai_response(response_text, os.path.basename(img_path))

                        # Scriere incrementală imediat după fiecare pagină procesată
                        if rows:
                            out_rows = []
                            for row in rows:
                                wav_file_name = f"{valid_count:012d}.wav"
                                out_rows.append([wav_file_name, row[1], row[2]])
                                valid_count += 1
                            writer.writerows(out_rows)
                            csv_file.flush()
                            extracted_segments += len(out_rows)

                        # Curățenie imediată imagine procesată
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass

                        progress.advance(task_pdf)

                # Reset descriere după PDF
                progress.update(task_pdf, description=f"[green]Procesare PDF cu Vision AI ({mode})...[/green]")

        finally:
            try:
                csv_file.close()
            except Exception:
                pass

    # Raportare finală (CSV-ul a fost scris incremental)
    if extracted_segments > 0:
        console.print(f"\n[bold green]Succes![/bold green] Au fost extrase {extracted_segments} segmente.")
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
