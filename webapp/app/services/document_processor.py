"""Document processing service for PDFs and text files."""
import os
import re
import csv
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from datetime import datetime

# Use pdfplumber for Windows compatibility (pure Python)
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# PyMuPDF as fallback (requires compilation)
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from app.config import PROJECTS_DIR


def clean_text(text: str) -> str:
    """Clean raw text extracted from PDFs or text files."""
    # Join hyphenated words at line breaks
    text = re.sub(r'-\n\s*', '', text)
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_sentences_simple(text: str) -> List[str]:
    """Simple sentence splitting without Spacy dependency."""
    # Split on sentence-ending punctuation followed by space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ])', text)
    return [s.strip() for s in sentences if s.strip()]


def filter_sentence(
    sentence: str, 
    mode: str,
    min_len: int = None,
    max_len: int = None,
    min_words: int = None
) -> bool:
    """Filter sentences based on mode criteria."""
    # Use defaults if not specified
    if min_len is None:
        min_len = 30
    if max_len is None:
        max_len = 100 if mode == 'TTS' else 220
    if min_words is None:
        min_words = 5 if mode == 'TTS' else 3
    
    # Length check
    if len(sentence) < min_len or len(sentence) > max_len:
        return False
    
    # Must start with uppercase
    if not sentence[0].isupper():
        return False
    
    # Must end with proper punctuation
    if sentence[-1] not in ['.', '!', '?']:
        return False
    
    # Word count check
    if len(sentence.split()) < min_words:
        return False
    
    # Filter abbreviation endings
    if any(sentence.endswith(suffix) for suffix in [" î.", " sec.", " vol.", " p."]):
        return False
    
    return True


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfplumber or PyMuPDF."""
    text = ""
    
    # Try pdfplumber first (pure Python, works on Windows)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            if not HAS_PYMUPDF:
                raise Exception(f"Error reading PDF {pdf_path}: {e}")
    
    # Fallback to PyMuPDF if available
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {e}")
    
    raise Exception("No PDF library available. Install pdfplumber: pip install pdfplumber")


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file (PDF or TXT)."""
    path = Path(file_path)
    if path.suffix.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def process_documents_to_csv(
    file_paths: List[str],
    project_folder: str,
    mode: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    min_len: int = None,
    max_len: int = None,
    min_words: int = None
) -> Tuple[int, str]:
    """
    Process document files and generate metadata.csv.
    
    Args:
        file_paths: List of paths to PDF/TXT files
        project_folder: Destination folder for the project
        mode: 'TTS' or 'STT'
        progress_callback: Optional callback for progress updates (progress%, message)
        min_len: Minimum sentence length (chars)
        max_len: Maximum sentence length (chars)
        min_words: Minimum words per sentence
    
    Returns:
        Tuple of (valid_count, csv_path)
    """
    os.makedirs(project_folder, exist_ok=True)
    
    all_text = ""
    
    # Extract text from all files
    total_files = len(file_paths)
    for i, file_path in enumerate(file_paths):
        if progress_callback:
            progress_callback(int((i / total_files) * 50), f"Extracting text from {Path(file_path).name}")
        
        try:
            text = extract_text_from_file(file_path)
            all_text += text + "\n"
        except Exception as e:
            if progress_callback:
                progress_callback(int((i / total_files) * 50), f"Error: {e}")
    
    # Clean and split text
    if progress_callback:
        progress_callback(50, "Cleaning and splitting text...")
    
    cleaned_text = clean_text(all_text)
    sentences = split_into_sentences_simple(cleaned_text)
    
    # Filter and write to CSV
    csv_path = os.path.join(project_folder, 'metadata.csv')
    
    # Find the next available index by checking existing files and CSV
    import glob
    next_index = 0
    
    # Check existing wav files in folder
    existing_wavs = glob.glob(os.path.join(project_folder, "*.wav"))
    for wav_file in existing_wavs:
        filename = os.path.basename(wav_file)
        try:
            idx = int(filename.replace('.wav', ''))
            next_index = max(next_index, idx + 1)
        except ValueError:
            pass
    
    # Also check CSV for highest index
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    wav_name = line.split('|')[0].strip()
                    try:
                        idx = int(wav_name.replace('.wav', ''))
                        next_index = max(next_index, idx + 1)
                    except ValueError:
                        pass
    
    valid_count = next_index
    entries_added = 0
    
    # Append mode if file exists, otherwise write new
    file_mode = 'a' if os.path.exists(csv_path) else 'w'
    
    with open(csv_path, file_mode, encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        
        total_sentences = len(sentences)
        for i, sentence in enumerate(sentences):
            if progress_callback and i % 100 == 0:
                progress_callback(50 + int((i / total_sentences) * 50), f"Filtering sentences... {i}/{total_sentences}")
            
            if filter_sentence(sentence, mode, min_len, max_len, min_words):
                # Clean up sentence
                sentence = sentence.replace("\n", " ").replace("\t", " ")
                
                # Generate filename
                wav_filename = f"{valid_count:012d}.wav"
                
                # Write: filename|original|original (normalized later)
                writer.writerow([wav_filename, sentence, sentence])
                valid_count += 1
                entries_added += 1
    
    if progress_callback:
        progress_callback(100, f"Done! Added {entries_added} sentences (total: {valid_count}).")
    
    return entries_added, csv_path


def cleanse_csv(project_folder: str) -> Tuple[int, int]:
    """
    Clean metadata.csv by removing entries without corresponding audio files.
    
    Returns:
        Tuple of (valid_count, removed_count)
    """
    csv_path = os.path.join(project_folder, 'metadata.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metadata.csv not found in {project_folder}")
    
    # Read existing CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        rows = list(reader)
    
    # Filter rows with existing audio
    valid_rows = []
    removed_count = 0
    
    for row in rows:
        if len(row) < 1:
            continue
        wav_path = os.path.join(project_folder, row[0])
        if os.path.exists(wav_path):
            valid_rows.append(row)
        else:
            removed_count += 1
    
    # Backup original
    backup_path = os.path.join(project_folder, 'metadata_original.csv')
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.rename(csv_path, backup_path)
    
    # Write cleaned CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(valid_rows)
    
    return len(valid_rows), removed_count

