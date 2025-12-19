"""Vision AI processor for extracting text from PDFs with OCR/Vision models."""
import os
import io
import csv
import base64
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Generator

# Try to import PDF libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

from app.config import (
    AI_PROVIDER,
    OLLAMA_MODEL_NAME,
    OPENAI_MODEL_NAME,
    OPENAI_API_KEY,
)


# System prompts for AI
SYSTEM_PROMPT_TTS = """Ești un expert în transcrierea și normalizarea textului din documente pentru limba română.

Returnează STRICT CSV cu separator | în format: TextBrut|TextNormalizat.
- TextBrut: textul exact cum apare în document
- TextNormalizat: textul pregătit pentru TTS (numerele în cuvinte, abrevieri expandate)
NU include ID-uri sau alte coloane."""

SYSTEM_PROMPT_STT = """Ești un expert în transcrierea textului pentru Whisper.

Returnează STRICT CSV cu separator | în format: TextBrut|TextNormalizat.
- TextBrut: textul exact cum apare în document  
- TextNormalizat: textul pregătit pentru STT (păstrează numerele, corectează punctuația)
NU include ID-uri sau alte coloane."""


def _load_image_bytes(image_path: str, max_dim: int = 1280) -> bytes:
    """Load and resize image for API consumption."""
    try:
        from PIL import Image
        
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
        with open(image_path, 'rb') as f:
            return f.read()


def _encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def iter_pdf_page_images(
    pdf_path: str,
    temp_dir: str,
    dpi: int = 200
) -> Generator[Tuple[int, str], None, None]:
    """Generate page images from PDF one at a time."""
    os.makedirs(temp_dir, exist_ok=True)
    
    # Try PyMuPDF first (best quality)
    if HAS_PYMUPDF:
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
        return
    
    # Fallback: pdf2image (requires poppler)
    if HAS_PDF2IMAGE:
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            for page_num, image in enumerate(images):
                image_path = os.path.join(temp_dir, f"page_{page_num:04d}.png")
                image.save(image_path, 'PNG')
                yield page_num, image_path
            return
        except Exception:
            pass
    
    raise RuntimeError(
        "Vision AI requires PyMuPDF or pdf2image. "
        "Install with: pip install pymupdf or use Docker."
    )


def process_image_with_ollama(image_path: str, mode: str) -> str:
    """Send image to Ollama and get CSV response."""
    import ollama
    
    system_prompt = SYSTEM_PROMPT_TTS if mode == "TTS" else SYSTEM_PROMPT_STT
    
    user_message = (
        "Analizează această imagine și extrage TOT textul relevant. "
        "Împarte în propoziții: o propoziție pe linie. "
        "Returnează DOAR liniile CSV în format: TextBrut|TextNormalizat."
    )
    
    try:
        image_bytes = _load_image_bytes(image_path, max_dim=1280)
        
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message, 'images': [image_bytes]}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return ""


def process_image_with_openai(image_path: str, mode: str) -> str:
    """Send image to OpenAI and get CSV response."""
    from openai import OpenAI
    
    system_prompt = SYSTEM_PROMPT_TTS if mode == "TTS" else SYSTEM_PROMPT_STT
    
    user_message = (
        "Analizează această imagine și extrage TOT textul relevant. "
        "Împarte în propoziții: o propoziție pe linie. "
        "Returnează DOAR liniile CSV în format: TextBrut|TextNormalizat."
    )
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        base64_image = _encode_image_to_base64(image_path)
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        return ""


def process_image(image_path: str, mode: str) -> str:
    """Process image with configured AI provider."""
    if AI_PROVIDER == "openai":
        return process_image_with_openai(image_path, mode)
    return process_image_with_ollama(image_path, mode)


def parse_ai_response(response_text: str) -> List[Tuple[str, str]]:
    """Parse AI response and extract valid CSV rows."""
    lines = response_text.strip().split('\n')
    valid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line or '|' not in line:
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 2:
            continue
        
        if len(parts) == 2:
            raw_text, norm_text = parts[0], parts[1]
        else:
            raw_text, norm_text = parts[-2], parts[-1]
        
        if len(raw_text) > 5:
            valid_rows.append((raw_text, norm_text))
    
    return valid_rows


def _count_pdf_pages(pdf_path: str) -> int:
    """Count pages in PDF using available library."""
    if HAS_PYMUPDF:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    
    if HAS_PDF2IMAGE:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception:
            pass
    
    # Default estimate
    return 10


def process_pdf_with_vision(
    pdf_path: str,
    project_folder: str,
    mode: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[int, str]:
    """
    Process PDF with Vision AI and create dataset.
    
    Args:
        pdf_path: Path to PDF file
        project_folder: Destination folder for the project
        mode: 'TTS' or 'STT'
        progress_callback: Optional callback for progress updates
    
    Returns:
        Tuple of (valid_count, csv_path)
    """
    # Check if we have required libraries
    if not HAS_PYMUPDF and not HAS_PDF2IMAGE:
        raise RuntimeError(
            "Vision AI requires PyMuPDF or pdf2image. "
            "Use Docker for full functionality, or install: pip install pymupdf"
        )
    
    os.makedirs(project_folder, exist_ok=True)
    temp_dir = os.path.join(project_folder, "temp_vision")
    
    csv_path = os.path.join(project_folder, 'metadata.csv')
    valid_count = 0
    
    # Get starting index if CSV exists
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter='|'):
                if row and row[0].endswith('.wav'):
                    idx = int(row[0].replace('.wav', ''))
                    valid_count = max(valid_count, idx + 1)
    
    # Count total pages
    total_pages = _count_pdf_pages(pdf_path)
    
    with open(csv_path, 'a', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        
        for page_num, img_path in iter_pdf_page_images(pdf_path, temp_dir, dpi=200):
            if progress_callback:
                progress_callback(
                    int((page_num / total_pages) * 100),
                    f"Processing page {page_num + 1}/{total_pages}..."
                )
            
            # Process with Vision AI
            response_text = process_image(img_path, mode)
            rows = parse_ai_response(response_text)
            
            # Write rows to CSV
            for raw_text, norm_text in rows:
                wav_filename = f"{valid_count:012d}.wav"
                writer.writerow([wav_filename, raw_text, norm_text])
                valid_count += 1
            
            csv_file.flush()
            
            # Cleanup image
            try:
                os.remove(img_path)
            except:
                pass
    
    # Cleanup temp dir
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    if progress_callback:
        progress_callback(100, f"Done! {valid_count} segments extracted.")
    
    return valid_count, csv_path

