from rich import print
from rich.console import Console
from rich.table import Table
import glob
import os
import sys
import csv
import re

def normalize_stt(text):
	# Pentru STT (Whisper), vrem "Standard Written Output".
	# Pastram numerele (10, 1990).
	# Pastram punctuatia (. , ? !).
	# Corectam doar spatierea si caracterele ciudate.
	
	# 1. Corectare spatii inainte de punctuatie
	text = re.sub(r'\s+([.,?!])', r'\1', text)
	
	# 2. Asigurare spatiu dupa punctuatie
	text = re.sub(r'([.,?!])(?=[a-zA-Z])', r'\1 ', text)
	
	# 3. Standardizare ghilimele
	text = text.replace('“', '"').replace('„', '"').replace('»', '"').replace('«', '"')
	
	return text.strip()

if __name__ == '__main__':
	console = Console()
	table = Table()
	table.add_column("STT Text Normalizer", style="cyan")
	table.add_row("Normalizes text for STT (Whisper).")
	table.add_row("Keeps numbers and punctuation. Fixes spacing.")
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
		
	out_path = os.path.join(project_folder, 'metadata_normalized_stt.csv')
	with open(out_path, 'w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f, delimiter='|')
		for row in data:
			if len(row) >= 2:
				original = row[1]
				normalized = normalize_stt(original)
				writer.writerow([row[0], original, normalized])
				
	console.print(f"Done. Saved to {out_path}")
