from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table

import pdfplumber
import spacy

import glob
import os
import re
import sys

from datetime import datetime, timedelta

def curata_text(text):
    # 1. Unim cuvintele despărțite la capăt de rând
    text = re.sub(r'-\n\s*', '', text)
    # 2. Înlocuim newline-urile din interiorul paragrafelor cu spațiu
    text = text.replace('\n', ' ')
    # 3. Eliminăm spațiile multiple
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == '__main__':

	console = Console()
		
	table = Table()
	table.add_column("STT CSV Generator (Romanian)", style="cyan")
	table.add_row("Extracts texts for Speech-to-Text (Whisper) training.")
	table.add_row("Splits into longer chunks (30-200 chars).")
	table.add_row("Does NOT normalize text (keeps numbers, punctuation).")
	
	console.print(table)
	
	app_folder = os.path.dirname(os.path.realpath(__file__))
	
	now = datetime.now()
	project_folder = os.path.join(app_folder, 'project_STT_' + now.strftime("%d%m%Y_%H%M%S"))
	console.print("Please select a [red]project folder[/red] (default [i]%s[/i])." % project_folder)
	in_project_folder = input()
	if not in_project_folder:
		in_project_folder = project_folder
	project_folder = in_project_folder
	if not os.path.exists(project_folder):
		os.mkdir(project_folder)
		
	console.print("Project folder is %s" % project_folder)
	
	# Load Spacy Model
	try:
		console.print("Loading Spacy model [green]ro_core_news_lg[/green]...")
		nlp = spacy.load("ro_core_news_lg")
		if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
			nlp.add_pipe('sentencizer')
	except OSError:
		console.print("[red]Error: Model 'ro_core_news_lg' not found.[/red]")
		sys.exit(1)

	# Select file types to read
	console.print("Please select the file types you want to load (t=text, p=pdf, tp=both) (default [i]tp[/i])")
	in_file_types = input()
	if not in_file_types:
		in_file_types = "tp"
	if not 't' in in_file_types and not 'p' in in_file_types:
		in_file_types = "tp"
		
	all_texts = ''

	if 't' in in_file_types:
		text_files = glob.glob(os.path.join(app_folder, 'texts', '*.txt'))
		console.print("Found %d text files" % (len(text_files)))
		for tf in text_files:
			try:
				with open(tf, "r", encoding= 'utf-8') as f:
					all_texts += f.read() + '\n'
			except Exception as e:
				console.print(f"[red]Error reading text file {tf}: {e}[/red]")
		
	if 'p' in in_file_types:
		pdf_files = glob.glob(os.path.join(app_folder, 'texts', '*.pdf'))
		console.print("Found %d pdf files" % (len(pdf_files)))
		for pdf_file in track(pdf_files, description="Reading PDFs..."):
			try:
				with pdfplumber.open(pdf_file) as pdf:
					for page in pdf.pages:
						page_text = page.extract_text()
						if page_text:
							all_texts +=  page_text + '\n'
			except Exception as e:
				console.print(f"[red]Error reading PDF {pdf_file}: {e}[/red]")
					
	console.print("Cleaning text...")
	all_texts = curata_text(all_texts)

	split_text = [all_texts[i:i+100000] for i in range(0, len(all_texts), 100000)]
	all_sentences = []
	
	console.print(f"Splitting text into sentences...")
	for st in track(split_text, description="Processing text chunks..."):
		doc = nlp(st)
		all_sentences.extend([str(sent).strip() for sent in doc.sents])

	csv_file_name = 'metadata.csv'
	csv_file_path = os.path.join(project_folder, csv_file_name)
	csv_file = open(csv_file_path, 'a', encoding = 'utf-8')
	
	valid_count = 0
	
	console.print("Filtering and writing to CSV...")
	for index, sentence in enumerate(all_sentences):
		
		# --- FILTERS FOR STT (Relaxed) ---
		
		# 1. Lungime: 30 - 200 caractere (3 - 30 secunde)
		if len(sentence) < 30 or len(sentence) > 220:
			continue
			
		# 2. Majuscula la inceput (important pentru Whisper)
		if not sentence[0].isupper():
			continue
			
		
		# 4. Minim 3 cuvinte (mai relaxat decat TTS)
		if len(sentence.split()) < 3:
			continue

		# --- END FILTERS ---

		sentence = sentence.replace("\n", " ")
		sentence = sentence.replace("\t", " ")
		
		# because we want the model to learn standard written output.
		cleansed_sentence = sentence
		
		wav_file_name = (str(valid_count) + '.wav').rjust(12, '0')
		
		csv_file.write(wav_file_name + "|" + sentence + "|" + cleansed_sentence + '\n')
		valid_count += 1
	
	csv_file.close()		
	
	# Estimate duration (avg 10s for STT chunks maybe?)
	duration_in_seconds = valid_count * 10 
	duration = timedelta(seconds=duration_in_seconds)
	
	console.print("[green]Success![/green]")
	console.print("%d valid sentences written to %s." % (valid_count, csv_file_name))
	console.print("Estimated audio duration: %s" % duration)
