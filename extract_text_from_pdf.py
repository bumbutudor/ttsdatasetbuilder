import os
import fitz  # PyMuPDF
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from google import genai
from google.genai import types

# ==========================================
# CONFIGURARE
# ==========================================

MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

SYSTEM_PROMPT = """
Ești un expert lingvistic specializat în curățarea datelor pentru antrenarea modelelor Speech-to-Text (STT). 
Analizează imaginea și extrage DOAR textul care reprezintă vorbire naturală, narațiune, dialoguri sau propoziții informative complete.

REGULI STRICTE DE EXCLUDERE (CE NU TREBUIE SĂ EXTRAGI):
1. NU extrage titluri, antete, subsoluri, numere de pagină sau numele revistei/cărții.
2. NU extrage exerciții de tip test grilă, liste de opțiuni (A, B, C), enunțuri de exerciții scurte ("Calculează:", "Alege varianta:").
3. NU extrage tabele, liste de prețuri, chei de răspunsuri.
4. Ignoră orice text care nu ar avea sens dacă ar fi citit cu voce tare într-un audiobook.

REGULI DE INCLUZIUNE (CE TREBUIE SĂ EXTRAGI):
1. Extrage doar propoziții complete (subiect + predicat) și dialoguri între personaje.
2. Dacă există o poveste sau o descriere, extrage-o integral.

REGULA CRITICĂ PENTRU PAGINI GOALE SAU IRELEVANTE:
Dacă pagina conține DOAR elemente excluse (exerciții grilă, tabele, titluri, imagini fără poveste, pagini goale) și NU conține nicio propoziție validă pentru STT, 
trebuie să răspunzi cu un singur cuvânt: NOT_FOUND

FORMATARE:
- Dacă găsești text valid, returnează-l ca un singur flux continuu (un singur paragraf).
- Nu folosi rânduri noi (\n).
- Scrie propoziție după propoziție, separate doar prin spațiu.
"""

USER_PROMPT = "Extrage textul din această pagină conform instrucțiunilor. Dacă nu există text valid, scrie NOT_FOUND."

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

def get_processed_files(log_path):
    """Citește fișierul de log pentru a vedea ce PDF-uri au fost finalizate."""
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                processed.add(line.strip())
    return processed

def mark_file_processed(log_path, filename):
    """Adaugă numele fișierului PDF în log-ul de procesare."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

def process_page_with_gemini(client, image_bytes):
    """Trimite imaginea paginii la Gemini și returnează textul extras."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1, 
            ),
            contents=[
                USER_PROMPT,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                )
            ]
        )
        
        if response.text:
            clean_text = response.text.replace('\n', ' ').strip()
            
            # Verificăm dacă modelul a returnat NOT_FOUND
            if "NOT_FOUND" in clean_text:
                return ""
            
            # Verificare suplimentară: dacă textul e prea scurt sau pare a fi o eroare de model
            if len(clean_text) < 5: 
                return ""

            return clean_text
        return ""
    except Exception as e:
        console.print(f"[red]Eroare API la procesarea paginii: {e}[/red]")
        return ""

def main():
    console.print(Panel.fit("[bold blue]PDF to STT Text Extractor[/bold blue]\n[dim]Folosind Gemini Vision[/dim]"))

    # 1. Input utilizator
    source_folder_str = Prompt.ask("Introdu calea către dosarul cu fișiere [bold]PDF[/bold]")
    output_file_str = Prompt.ask("Introdu calea completă către fișierul [bold]TXT[/bold] de ieșire (ex: C:\\Date\\output.txt)")

    source_path = Path(source_folder_str)
    output_path = Path(output_file_str)

    # Validări
    if not source_path.exists() or not source_path.is_dir():
        console.print(f"[bold red]Eroare:[/bold red] Folderul sursă '{source_folder_str}' nu există.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.parent / "processed_pdfs.log"

    # 2. Setup
    client = get_api_client()
    pdf_files = sorted(list(source_path.glob("*.pdf")))
    
    if not pdf_files:
        console.print("[yellow]Nu s-au găsit fișiere PDF în folderul specificat.[/yellow]")
        return

    # 3. Logica de Resume
    processed_pdfs = get_processed_files(log_path)
    files_to_process = [f for f in pdf_files if f.name not in processed_pdfs]

    if not files_to_process:
        console.print("[bold green]Toate fișierele PDF din acest dosar au fost deja procesate![/bold green]")
        return

    console.print(f"\n[cyan]Am găsit {len(pdf_files)} fișiere. {len(processed_pdfs)} deja procesate.[/cyan]")
    console.print(f"[bold green]Urmează să procesăm {len(files_to_process)} fișiere.[/bold green]\n")

    # 4. Procesarea
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        total_task = progress.add_task("[bold green]Progres Total PDF-uri", total=len(files_to_process))
        
        with open(output_path, 'a', encoding='utf-8') as outfile:
            
            for pdf_file in files_to_process:
                try:
                    doc = fitz.open(pdf_file)
                    num_pages = len(doc)
                    
                    page_task = progress.add_task(f"Procesez: {pdf_file.name}", total=num_pages)
                    
                    file_text_buffer = []

                    for page_num in range(num_pages):
                        page = doc.load_page(page_num)
                        
                        mat = fitz.Matrix(2, 2) 
                        pix = page.get_pixmap(matrix=mat)
                        img_bytes = pix.tobytes("png")

                        text_page = process_page_with_gemini(client, img_bytes)
                        
                        # Adăugăm textul doar dacă nu este gol (NOT_FOUND returnează "")
                        if text_page:
                            file_text_buffer.append(text_page)

                        progress.advance(page_task)

                    # Scriem în fișier doar dacă am extras ceva din tot PDF-ul
                    if file_text_buffer:
                        full_text = " ".join(file_text_buffer) + " "
                        outfile.write(full_text)
                        outfile.flush()
                    
                    mark_file_processed(log_path, pdf_file.name)
                    
                    progress.remove_task(page_task)
                    progress.advance(total_task)
                    
                    doc.close()

                except Exception as e:
                    console.print(f"\n[bold red]Eroare critică la fișierul {pdf_file.name}: {e}[/bold red]")
                    continue

    console.print(f"\n[bold green]Procesare completă![/bold green]")
    console.print(f"Textul a fost salvat în: [underline]{output_path}[/underline]")
    console.print(f"Log-ul procesării este în: {log_path}")

if __name__ == "__main__":
    main()