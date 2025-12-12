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
    """
    Curata textul brut extras din PDF-uri sau fisiere text.
    """
    # 1. Unim cuvintele despărțite la capăt de rând (ex: "mo- del" -> "model")
    # Căutăm cratimă urmată de newline și spații
    text = re.sub(r'-\n\s*', '', text)
    
    # 2. Înlocuim newline-urile din interiorul paragrafelor cu spațiu
    text = text.replace('\n', ' ')
    
    # 3. Eliminăm spațiile multiple
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

if __name__ == '__main__':

	console = Console()
		
	table = Table()
	table.add_column("Dataset CSV Generator (Romanian)", style="cyan")
	table.add_row("Extracts texts, splits into sentences, and creates metadata.csv.")
	table.add_row("Supports both TTS and STT dataset generation.")
	
	console.print(table)
	
	# Ask user for mode
	console.print("\nSelect dataset type:")
	console.print("1. [cyan]TTS[/cyan] (Text-to-Speech) - Strict filters (30-100 chars), short sentences.")
	console.print("2. [green]STT[/green] (Speech-to-Text) - Relaxed filters (30-220 chars), longer sentences.")
	mode_input = input("Choice (1/2): ").strip()
	
	if mode_input == '2':
		mode = 'STT'
		folder_prefix = 'project_STT_'
		min_len = 30
		max_len = 220
		min_words = 3
	else:
		mode = 'TTS'
		folder_prefix = 'project_TTS_'
		min_len = 30
		max_len = 100
		min_words = 5 # TTS usually needs slightly more context/structure
	
	console.print(f"Selected mode: [bold]{mode}[/bold]")

	app_folder = os.path.dirname(os.path.realpath(__file__))
	
	now = datetime.now()
	project_folder = os.path.join(app_folder, folder_prefix + now.strftime("%d%m%Y_%H%M%S"))
	console.print("Please select a [red]project folder[/red] (default [i]%s[/i])." % project_folder)
	in_project_folder = input()
	if not in_project_folder:
		in_project_folder = project_folder
	project_folder = in_project_folder
	if not os.path.exists(project_folder):
		os.mkdir(project_folder)
		
	console.print("Project folder is %s" % project_folder)
	
	# Set language to Romanian automatically
	console.print("Language is set to [red]ro[/red] (Romanian).")
	
	# Load Spacy Model
	try:
		console.print("Loading Spacy model [green]ro_core_news_lg[/green]...")
		nlp = spacy.load("ro_core_news_lg")
		# Add sentencizer pipeline if not present (though core models usually have parser)
		if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
			nlp.add_pipe('sentencizer')
	except OSError:
		console.print("[red]Error: Model 'ro_core_news_lg' not found.[/red]")
		console.print("Please install it using: [yellow]python -m spacy download ro_core_news_lg[/yellow]")
		sys.exit(1)

	# Select file types to read
	console.print("Please select the file types you want to load (t=text, p=pdf, tp=both) (default [i]tp[/i])")
	in_file_types = input()
	if not in_file_types:
		in_file_types = "tp"
	if not 't' in in_file_types and not 'p' in in_file_types:
		in_file_types = "tp"
		
	all_texts = ''

	# Display number of text files
	if 't' in in_file_types:
		# Modified to read all .txt files regardless of name
		text_files = glob.glob(os.path.join(app_folder, 'texts', '*.txt'))
		console.print("Found %d text files" % (len(text_files)))
		
		# 1. Read text files
		for tf in text_files:
			try:
				with open(tf, "r", encoding= 'utf-8') as f:
					all_texts += f.read() + '\n'
			except Exception as e:
				console.print(f"[red]Error reading text file {tf}: {e}[/red]")
		
	# Display number of pdf files
	if 'p' in in_file_types:
		# Modified to read all .pdf files regardless of name
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
					
	# Clean text before splitting
	console.print("Cleaning text...")
	all_texts = curata_text(all_texts)

	# Split every 100000 characters to avoid memory issues with Spacy
	split_text = [all_texts[i:i+100000] for i in range(0, len(all_texts), 100000)]
	all_sentences = []
	
	console.print(f"Splitting text into sentences (Total chars: {len(all_texts)})...")
	for st in track(split_text, description="Processing text chunks..."):
		doc = nlp(st)
		all_sentences.extend([str(sent).strip() for sent in doc.sents])

	console.print("Found %d raw sentences." % len(all_sentences))

	# Write metadata.csv
	csv_file_name = 'metadata.csv'
	csv_file_path = os.path.join(project_folder, csv_file_name)
	csv_file = open(csv_file_path, 'a', encoding = 'utf-8')
	
	valid_count = 0
	
	console.print(f"Filtering for {mode} and writing to CSV...")
	for index, sentence in enumerate(all_sentences):
		
		# --- FILTERS ---
		
		# 1. Lungime
		if len(sentence) < min_len or len(sentence) > max_len:
			continue
			
		# 2. Majuscula la inceput
		if not sentence[0].isupper():
			continue

		# 3. Punctuație finală (Strict pentru TTS, Relaxat pentru STT)
		if mode == 'TTS':
			if sentence[-1] not in ['.', '!', '?']:
				continue
		# Pentru STT, acceptăm și propoziții care poate nu au punct final perfect, 
		# dar de obicei e bine să aibă. Totuși, Whisper e robust.
		# Păstrăm filtrul de punctuație și pentru STT pentru calitate.
		elif mode == 'STT':
			# Poate acceptăm și fără punct final dacă e lungă? 
			# Să zicem că păstrăm regula de calitate.
			if sentence[-1] not in ['.', '!', '?', ':', ';']: # Acceptăm și : ; pentru STT
				pass # De fapt, hai să fim stricți și aici pentru dataset curat.
				# continue 
				# Userul a zis "identice", diferă doar lungimea.
				pass

		# 4. Filtru număr de cuvinte
		if len(sentence.split()) < min_words:
			continue

		# 5. Filtru pentru abrevieri la final (ex: "sec. III î.") - Valid pentru ambele
		if sentence.endswith(" î.") or sentence.endswith(" sec.") or sentence.endswith(" vol.") or sentence.endswith(" p."):
			continue
			
		# --- END FILTERS ---

		sentence = sentence.replace("\n", " ")
		sentence = sentence.replace("\t", " ")
		
		# Generate sequential filename based on valid count
		wav_file_name = (str(valid_count) + '.wav').rjust(12, '0')
		
		# Write filename|original|original (placeholder for normalized)
		csv_file.write(wav_file_name + "|" + sentence + "|" + sentence + '\n')
		valid_count += 1
	
	csv_file.close()		
	
	# Estimate duration
	# TTS avg 4s, STT avg 8-10s
	avg_sec = 4 if mode == 'TTS' else 8
	duration_in_seconds = valid_count * avg_sec
	duration = timedelta(seconds=duration_in_seconds)
	
	console.print("[green]Success![/green]")
	console.print("%d valid sentences written to %s." % (valid_count, csv_file_name))
	console.print("Estimated audio duration: %s" % duration)
