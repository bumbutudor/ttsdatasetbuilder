# TTS & STT Dataset Creator
This tool helps you to create datasets for Text-to-Speech (TTS) and Speech-to-Text (STT) tasks. It allows you to generate metadata from texts/PDFs, record your voice reading sentences, and normalize text for training models like Whisper or Tacotron.

## Installation

### Using pip
Install the required dependencies:
```bash
pip install -r requirements.txt
python -m spacy download ro_core_news_lg
```

## Usage
There are 6 applications in this repository:
main applications in this repository. Below is a guide on when and how to use each one.

### 1. main_generate_csv.py
**Use when:** You have raw text files (`.txt`, `.pdf`) in the `texts` folder and want to create a dataset structure.
**Description:**
- Scans the `texts` folder.
- Cleans and splits text into sentences using Spacy (Romanian).
- Filters sentences based on length and structure (Strict for TTS, Relaxed for STT).
- Generates a `metadata.csv` file in a new project folder.

### 2. main_generator.py
**Use when:** You want to record audio for the sentences in your `metadata.csv`.
**Description:**
- Reads `metadata.csv`.
- Displays sentences one by one.
- Records audio from your microphone.
- Trims silence automatically (Strict for TTS, Relaxed for STT).
- Saves `.wav` files.
**Controls:** `n` (next), `d` (discard/retry), `e` (exit).

### 3. main_vision_extractor.py
**Use when:** You have complex documents (manuals, textbooks with formulas) and want high-quality extraction using AI.
**Description:**
- Uses a Vision LLM (via Ollama) to "read" PDFs page-by-page.
- Extracts text and normalizes it simultaneously.
- **TTS Mode:** Expands math symbols, removes brackets, converts numbers to words.
- **STT Mode:** Keeps numbers, standardizes quotes and punctuation.
- **Prerequisites:** [Ollama](https://ollama.com/) running with the model specified in `vision_config.py`.

### 4. main_cleanse_csv.py
**Use when:** You have finished recording or manually deleted some bad audio files.
**Description:**
- Scans your project folder.
- Checks if every entry in `metadata.csv` has a corresponding `.wav` file.
- Removes entries where the audio is missing.
- Ensures your dataset is clean and ready for training.

### 5. main_normalize_text_STT.py
**Use when:** You have a dataset and want to prepare the text for **Speech-to-Text (Whisper)** training.
**Description:**
- Reads `metadata.csv`.
- Creates `metadata_normalized_stt.csv`.
- **Rules:** Keeps numbers (digits), standardizes quotes („”), preserves punctuation, removes citations.

### 6. main_normalize_text_TTS.py
**Use when:** You have a dataset and want to prepare the text for **Text-to-Speech** training.
**Description:**
- Reads `metadata.csv`.
- Creates `metadata_normalized_tts.csv`.
- **Rules:** Expands numbers to words, expands abbreviations (dl. -> domnul), expands math symbols (+ -> plus), removes bracket