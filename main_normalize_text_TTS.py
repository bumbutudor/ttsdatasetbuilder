from rich import print
from rich.console import Console
from rich.table import Table
import glob
import os
import sys
import csv
import re
try:
	from num2words import num2words
except ImportError:
	print("Please install num2words: pip install num2words")
	sys.exit(1)

# Dictionar de normalizare pentru TTS (extindere abrevieri si simboluri)
c_map_ro = {
	'€': 'euro',
	'$': 'dolari',
	'dl.': 'domnul',
	'dna.': 'doamna',
	'd-na': 'doamna',
	'd-ra': 'domnișoara',
	'etc.': 'et cetera',
	'pt.': 'pentru',
	'nr.': 'numărul',
	'str.': 'strada',
	'bl.': 'blocul',
	'sc.': 'scara',
	'ap.': 'apartamentul',
	'jud.': 'județul',
	'sec.': 'secolul',
	'%': 'la sută',
	'+': 'plus',
	'=': 'egal',
	'&': 'și',
	'“': '"',
	'„': '"',
	'»': '"',
	'«': '"'
}

def normalize_tts(text):
	# 1. Expandare abrevieri si simboluri
	for k, v in c_map_ro.items():
		if k.endswith('.'):
			k_esc = re.escape(k)
			# Replace whole word match for abbreviations ending in dot
			text = re.sub(r'(?<!\w)' + k_esc + r'(?!\w)', v, text)
		else:
			text = text.replace(k, v)
			
	# 2. Eliminare paranteze si continutul lor (optional, dar recomandat pentru TTS curat)
	# text = re.sub(r'\([^)]*\)', '', text)
	# text = re.sub(r'\[[^\]]*\]', '', text)
	
	# 3. Eliminare ghilimele (nu se aud)
	text = text.replace('"', '').replace("'", "")
	
	# 4. Expandare numere folosind num2words
	def replace_num(match):
		num_str = match.group(0)
		try:
			# Inlocuim virgula cu punct pentru parsing, daca e cazul
			clean_num = num_str.replace(',', '.')
			if '.' in clean_num:
				val = float(clean_num)
			else:
				val = int(clean_num)
			return num2words(val, lang='ro')
		except:
			return num_str

	# Cautam numere (intregi sau cu zecimale)
	text = re.sub(r'\b\d+([.,]\d+)?\b', replace_num, text)
	
	return text

if __name__ == '__main__':
	console = Console()
	table = Table()
	table.add_column("TTS Text Normalizer", style="cyan")
	table.add_row("Normalizes text for TTS: Expands abbreviations, removes silent punctuation.")
	console.print(table)
	
	app_folder = os.path.dirname(os.path.realpath(__file__))
	project_directories = glob.glob(os.path.join(app_folder, 'project*'))
	
	if not project_directories:
		console.print("No project folders found.")
		sys.exit(0)
		
	console.print("Select project folder:")
	for i, pd in enumerate(project_directories):
		console.print(f"{i}: {pd}")
	
	pid = input() or "0"
	project_folder = project_directories[int(pid)]
	
	csv_path = os.path.join(project_folder, 'metadata.csv')
	if not os.path.exists(csv_path):
		console.print("metadata.csv not found.")
		sys.exit(0)
		
	with open(csv_path, 'r', encoding='utf-8') as f:
		reader = csv.reader(f, delimiter='|')
		data = list(reader)
		
	out_path = os.path.join(project_folder, 'metadata_normalized_tts.csv')
	with open(out_path, 'w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f, delimiter='|')
		for row in data:
			if len(row) >= 2:
				original = row[1]
				normalized = normalize_tts(original)
				writer.writerow([row[0], original, normalized])
				
	console.print(f"Done. Saved to {out_path}")
	console.print("[yellow]Note: Number expansion (10 -> zece) requires 'num2words' library or manual implementation.[/yellow]")
