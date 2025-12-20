"""Vision AI processor for extracting text from PDFs with OCR/Vision models.

Adapted from main_vision_extractor.py to support Web/Docker environment.
Features:
- Robust image resizing to prevent OOM/500 errors
- Automatic retry logic for Ollama
- PyMuPDF (fitz) integration
"""
import os
import io
import csv
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Generator

# Logging configuration
logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF (fitz) not found. Vision processing will fail.")

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    logger.warning("Pillow (PIL) not found. Image resizing will be disabled.")

from app.config import (
    AI_PROVIDER,
    OLLAMA_MODEL_NAME,
    OPENAI_MODEL_NAME,
    OPENAI_API_KEY,
)


def _load_prompt_file(filename: str, fallback: str) -> str:
    """Load prompt from app/services/prompts/ folder."""
    try:
        # Obținem directorul unde se află scriptul curent (app/services)
        current_dir = Path(__file__).resolve().parent
        
        # Construim calea către folderul prompts (app/services/prompts/filename)
        prompts_path = current_dir / "prompts" / filename
        
        if prompts_path.exists():
            logger.info(f"Loading system prompt from: {prompts_path}")
            with open(prompts_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            return content if content else fallback
        else:
            logger.warning(f"Prompt file not found at: {prompts_path}. Using fallback.")
            
    except Exception as e:
        logger.warning(f"Could not load prompt from {filename}: {e}")
    
    return fallback


# Load system prompts
SYSTEM_PROMPT_TTS = _load_prompt_file(
    "system_tts.txt",
    """Ești un expert în transcrierea și normalizarea textului din documente pentru limba română.

Returnează STRICT CSV cu separator | în format: TextBrut|TextNormalizat.
- TextBrut: textul exact cum apare în document
- TextNormalizat: textul pregătit pentru TTS (numerele în cuvinte, abrevieri expandate)
NU include ID-uri sau alte coloane."""
)

SYSTEM_PROMPT_STT = _load_prompt_file(
    "system_stt.txt",
    """Ești un expert în transcrierea textului pentru Whisper.

Returnează STRICT CSV cu separator | în format: TextBrut|TextNormalizat.
- TextBrut: textul exact cum apare în document  
- TextNormalizat: textul pregătit pentru STT (păstrează numerele, corectează punctuația)
NU include ID-uri sau alte coloane."""
)


def _load_image_bytes(image_path: str, max_dim: int = 1280) -> bytes:
    """Load and resize image for API consumption (Ollama friendly)."""
    if HAS_PILLOW:
        try:
            with Image.open(image_path) as im:
                im = im.convert('RGB')
                w, h = im.size
                # Resize if larger than max_dim
                scale = min(1.0, float(max_dim) / float(max(w, h)))
                if scale < 1.0:
                    im = im.resize((int(w * scale), int(h * scale)))
                
                bio = io.BytesIO()
                # Save as JPEG (smaller than PNG, better for LLMs)
                im.save(bio, format='JPEG', quality=85, optimize=True)
                return bio.getvalue()
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {e}")
            
    # Fallback if Pillow missing or error
    with open(image_path, 'rb') as f:
        return f.read()


def _encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for OpenAI."""
    # We can reuse the resizing logic by getting bytes first, then encoding
    img_bytes = _load_image_bytes(image_path, max_dim=1280)
    return base64.b64encode(img_bytes).decode('utf-8')


def iter_pdf_page_images(
    pdf_path: str,
    temp_dir: str,
    dpi: int = 200
) -> Generator[Tuple[int, str], None, None]:
    """Generate page images from PDF one at a time using PyMuPDF."""
    os.makedirs(temp_dir, exist_ok=True)
    
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF is required. Please install it in the Dockerfile.")
        
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


def process_image_with_ollama(image_path: str, mode: str, model: Optional[str] = None) -> str:
    """Send image to Ollama with RETRY logic for OOM/500 errors."""
    import ollama
    
    system_prompt = SYSTEM_PROMPT_TTS if mode == "TTS" else SYSTEM_PROMPT_STT
    model_name = model or OLLAMA_MODEL_NAME
    
    user_message = (
        "Analizează această imagine (o pagină) și extrage TOT textul relevant. "
        "Împarte în propoziții/segmente: o propoziție pe linie. "
        "Returnează DOAR liniile CSV în format: TextBrut|TextNormalizat (2 coloane). "
        "NU include titluri sau markdown. "
        "NU folosi caracterul | în interiorul textului (doar ca separator între cele 2 coloane)."
    )
    
    # Try 1: Standard quality
    try:
        image_bytes = _load_image_bytes(image_path, max_dim=1280)
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message, 'images': [image_bytes]}
            ]
        )
        return response['message']['content']
        
    except Exception as e:
        msg = str(e)
        logger.warning(f"Ollama error (attempt 1): {msg}")
        
        # Try 2: Aggressive resize (if error was likely OOM/500)
        if '500' in msg or 'Internal Server Error' in msg or 'eof' in msg.lower():
            try:
                logger.info(f"Retrying with smaller image for {Path(image_path).name}...")
                image_bytes = _load_image_bytes(image_path, max_dim=896)
                
                response = ollama.chat(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message, 'images': [image_bytes]},
                    ]
                )
                return response['message']['content']
            except Exception as e2:
                logger.error(f"Ollama error (attempt 2): {e2}")
        
        return ""


def process_image_with_openai(image_path: str, mode: str, model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Send image to OpenAI API."""
    from openai import OpenAI
    
    system_prompt = SYSTEM_PROMPT_TTS if mode == "TTS" else SYSTEM_PROMPT_STT
    model_name = model or OPENAI_MODEL_NAME
    key = api_key or OPENAI_API_KEY
    
    user_message = (
        "Analizează această imagine (o pagină) și extrage TOT textul relevant. "
        "Împarte în propoziții/segmente: o propoziție pe linie. "
        "Returnează DOAR liniile CSV în format: TextBrut|TextNormalizat (2 coloane). "
        "NU include titluri sau markdown. "
        "NU folosi caracterul | în interiorul textului (doar ca separator între cele 2 coloane)."
    )
    
    try:
        client = OpenAI(api_key=key)
        base64_image = _encode_image_to_base64(image_path)
        
        api_params = {
            "model": model_name,
            "messages": [
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
            ]
        }
        
        # New models use max_completion_tokens
        is_new_model = any(x in model_name.lower() for x in ["gpt-5", "o1-", "o3-"])
        if is_new_model:
            api_params["max_completion_tokens"] = 4096
        else:
            api_params["max_tokens"] = 4096
            
        response = client.chat.completions.create(**api_params)
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return ""


def process_image(image_path: str, mode: str, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Dispatcher for image processing."""
    use_provider = provider or AI_PROVIDER
    if use_provider == "openai":
        return process_image_with_openai(image_path, mode, model, api_key)
    return process_image_with_ollama(image_path, mode, model)


def parse_ai_response(response_text: str) -> List[Tuple[str, str]]:
    """Parse AI response and extract valid CSV rows."""
    if not response_text:
        return []
        
    lines = response_text.strip().split('\n')
    valid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line or '|' not in line:
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 2:
            continue
        
        # Handle cases where AI might add an index column or file name
        if len(parts) == 2:
            raw_text, norm_text = parts[0], parts[1]
        else:
            # Fallback: take the last two columns
            raw_text, norm_text = parts[-2], parts[-1]
        
        # Basic validation
        if len(raw_text) > 2:
            valid_rows.append((raw_text, norm_text))
    
    return valid_rows


def process_pdf_with_vision(
    pdf_path: str,
    project_folder: str,
    mode: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> Tuple[int, str]:
    """
    Process PDF with Vision AI.
    """
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is missing. Vision features require it.")
    
    os.makedirs(project_folder, exist_ok=True)
    temp_dir = os.path.join(project_folder, "temp_vision")
    csv_path = os.path.join(project_folder, 'metadata.csv')
    
    # Determine start index
    valid_count = 0
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                for row in reader:
                    if row and row[0].endswith('.wav'):
                        try:
                            idx = int(row[0].replace('.wav', ''))
                            valid_count = max(valid_count, idx + 1)
                        except ValueError:
                            pass
        except Exception:
            pass
    
    # Count total pages (fast)
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        logger.error(f"Could not open PDF {pdf_path}: {e}")
        return 0, csv_path
    
    entries_added = 0
    
    # Process page by page
    try:
        # Use 'a' append mode
        with open(csv_path, 'a', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter='|')
            
            for page_num, img_path in iter_pdf_page_images(pdf_path, temp_dir, dpi=200):
                if progress_callback:
                    progress_callback(
                        int((page_num / total_pages) * 100),
                        f"Processing page {page_num + 1}/{total_pages}..."
                    )
                
                # Call AI
                response_text = process_image(img_path, mode, provider, model, api_key)
                rows = parse_ai_response(response_text)
                
                # Write rows immediately
                for raw_text, norm_text in rows:
                    wav_filename = f"{valid_count:012d}.wav"
                    writer.writerow([wav_filename, raw_text, norm_text])
                    valid_count += 1
                    entries_added += 1
                
                csv_file.flush()
                
                # Cleanup image immediately
                try:
                    os.remove(img_path)
                except:
                    pass
                    
    finally:
        # Cleanup temp dir
        try:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
            
    if progress_callback:
        progress_callback(100, f"Done! {entries_added} new segments extracted.")
        
    return entries_added, csv_path