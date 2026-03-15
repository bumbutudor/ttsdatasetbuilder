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
	
	# 0. Eliminare URL-uri
	text = re.sub(r'https?://\S+', '', text)

	# 1. Eliminare continut paranteze patrate (referinte bibliografice [12], [3])
	text = re.sub(r'\[.*?\]', '', text)

	# 2. Inlocuire paranteze rotunde cu virgule
	# E mai sigur pentru STT sa vada virgule decat paranteze pe care nu le poate "auzi"
	text = text.replace('(', ', ').replace(')', ', ')

	# 3. Standardizare linii de pauza
	# En-dash si Em-dash devin " – " (spatiu en-dash spatiu)
	# Folosim regex pentru a gestiona spatiile din jur
	text = re.sub(r'\s*[–—]\s*', ' – ', text)

	# 4. Standardizare ghilimele la formatul romanesc „text”
	# Inlocuim formele explicite (franceze/englezesti) cu cele romanesti
	text = text.replace('«', '„').replace('»', '”')
	text = text.replace('“', '„')
	# Nota: ” (U+201D) este deja closing quote corect in romana, il pastram.

	# Gestionare ghilimele drepte "
	# Daca e la inceput de cuvant (precedat de spatiu sau inceput de linie), e opening „
	text = re.sub(r'(^|\s)"', r'\1„', text)
	# Orice alt " ramas este considerat closing ”
	text = text.replace('"', '”')
	
	# 5. Corectare spatii si punctuatie
	# Eliminam spatii duble
	text = re.sub(r'\s+', ' ', text)
	# Eliminam spatiu inainte de semne de punctuatie
	text = re.sub(r'\s+([.,?!:;])', r'\1', text)
	# Asiguram spatiu dupa semne de punctuatie (daca urmeaza litera)
	# Nu punem spatiu daca urmeaza cifra (ex: 3.14, 1,2)
	text = re.sub(r'([.,?!:;])(?=[a-zA-Z])', r'\1 ', text)

	# Curatare virgule duble sau virgula-punct rezultate din inlocuiri
	text = re.sub(r',\s*,', ', ', text)
	text = re.sub(r',\s*\.', '.', text)
	
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
