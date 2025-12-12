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
	# Simboluri
	'€': 'euro',
	'$': 'dolari',
	'%': 'la sută',
	'+': 'plus',
	'=': 'egal',
	'&': 'și',
	'“': '"',
	'„': '"',
	'»': '"',
	'«': '"',
	
	# Titluri, Persoane si Adresare
	'dl.': 'domnul',
	'dna.': 'doamna',
	'd-na': 'doamna',
	'd-ra': 'domnișoara',
	'dra.': 'domnișoara',
	'dvs.': 'dumneavoastră',
	'dlui': 'domnului',
	'dnei': 'doamnei',
	'dr.': 'doctor',
	'prof.': 'profesor',
	'ing.': 'inginer',
	'arh.': 'arhitect',
	'asist.': 'asistent',
	'pr.': 'preotul',
	'Sf.': 'Sfântul',
	'sf.': 'sfântul',
	'Sf.a': 'Sfânta',

	# Adrese, Locatii si Urbanism
	'str.': 'strada',
	'b-dul': 'bulevardul',
	'bd.': 'bulevardul',
	'sos.': 'șoseaua',
	'al.': 'aleea',
	'fdt.': 'fundătura',
	'nr.': 'numărul',
	'bl.': 'blocul',
	'sc.': 'scara',
	'et.': 'etajul',
	'ap.': 'apartamentul',
	'sect.': 'sectorul',
	'jud.': 'județul',
	'mun.': 'municipiul',
	'loc.': 'localitatea',
	'cod.': 'codul',

	# Expresii Comune si Academice
	'etc.': 'și așa mai departe',
	'ș.a.': 'și altele',
	'Ș.a.': 'și altele',
	'ș.a.m.d.': 'și așa mai departe',
	'Ș.a.m.d.': 'și așa mai departe',
	'ș.c.l.': 'și celelalte',
	'Ș.c.l.': 'și celelalte',
	'ex.': 'exemplu',
	'cca.': 'circa',
	'aprox.': 'aproximativ',
	'pt.': 'pentru',
	'ptr.': 'pentru',
	'vs.': 'versus',
	'a.c.': 'anul curent',
	'pag.': 'pagina',
	'p.': 'pagina',
	'vol.': 'volumul',
	'art.': 'articolul',
	'cap.': 'capitolul',
	'alin.': 'alineatul',
	'ed.': 'editura',

	# Timp si Calendar
	'sec.': 'secolul',
	'ian.': 'ianuarie',
	'feb.': 'februarie',
	'mar.': 'martie',
	'apr.': 'aprilie',
	'iun.': 'iunie',
	'iul.': 'iulie',
	'aug.': 'august',
	'sept.': 'septembrie',
	'oct.': 'octombrie',
	'nov.': 'noiembrie',
	'dec.': 'decembrie',
	'hr.': 'ora',
	'min.': 'minute',
	'î.Hr.': 'înainte de Hristos',
	'Î.Hr.': 'înainte de Hristos',
	'ÎdC': 'înainte de Hristos',
	'Î.d.Hr.': 'înainte de Hristos',
	'e.n.': 'era noastră',
	'd.Hr.': 'după Hristos',
	'D.Hr.': 'după Hristos',

	# Acronime Majore - Citite ca un cuvant
	'NATO': 'nato',
	'UNESCO': 'unesco',
	'ONU': 'onu',
	'TAROM': 'tarom',
	'NASA': 'nasa',
	'SIDA': 'sida',
	'COVID': 'covid',
	'FIFA': 'fifa',
	'UEFA': 'uefa',

	# Acronime Majore - Citite pe litere
	'SUA': 'S U A',
	'UE': 'U E',
	'SRI': 'S R I',
	'DNA': 'D N A',
	'ONG': 'O N G',
	'TV': 'te ve',
	'CD': 'ce de',
	'PC': 'pe ce',
	'CV': 'ci vi',
	'WC': 've ce',
	'OK': 'ochei',
	'SOS': 'se o se',

	# Specific Republica Moldova
	'R. Moldova': 'Republica Moldova',
	'RM': 'Republica Moldova',
	'mun. Chișinău': 'municipiul Chișinău',
	'ASEM': 'ASEM',
	'ULIM': 'ULIM',
	'USM': 'U S M'
}

def roman_to_int(s):
	rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
	int_val = 0
	for i in range(len(s)):
		if s[i] not in rom_val:
			return 0
		if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
			int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
		else:
			int_val += rom_val[s[i]]
	return int_val

def normalize_tts(text):
	# 0. Eliminare URL-uri
	text = re.sub(r'https?://\S+', '', text)

	# 1. Eliminare continut paranteze patrate (referinte bibliografice [12], [3])
	text = re.sub(r'\[.*?\]', '', text)

	# 2. Expandare abrevieri si simboluri din dictionar
	for k, v in c_map_ro.items():
		if k.endswith('.'):
			k_esc = re.escape(k)
			# Replace whole word match for abbreviations ending in dot
			text = re.sub(r'(?<!\w)' + k_esc + r'(?!\w)', v, text)
		else:
			text = text.replace(k, v)
			
	# --- NORMALIZARE AVANSATA SIMBOLURI (Matematica, Fizica, Punctuație) ---

	# 3.1 Mulțimi specifice (N*, R*, Z*, Q*) -> N stelat
	text = re.sub(r'([NRZQ])\*', r'\1 stelat', text)

	# 3.2 Gradele Celsius/Fahrenheit
	text = text.replace("°C", " grade Celsius")
	text = text.replace("°F", " grade Fahrenheit")
	text = text.replace("°", " grade")

	# 3.3 Unități de măsură compuse
	text = text.replace("km/h", " kilometri pe oră")
	text = text.replace("m/s", " metri pe secundă")

	# 3.4 Minus vs Cratimă
	# Cratima rămâne cratimă doar între litere (ex: s-a, teoretic-fizic).
	# În rest (între cifre, spații, sau unar), devine "minus".
	text = re.sub(r'(?<=\d)-(?=\d)', ' minus ', text)   # 5-2
	text = re.sub(r'(?<=\s)-(?=\d)', ' minus ', text)   # -5 (unar)
	text = re.sub(r'(?<=\d)-(?=\s)', ' minus ', text)   # 5- 
	text = text.replace("−", " minus ") # Simbolul matematic real de minus

	# 3.5 Împărțire vs Două puncte
	# : devine "împărțit la" doar între cifre (pentru operatii matematice)
	# ATENTIE: Conflict cu Ora/Scor (12:30). 
	# Regula anterioara (12:30 -> 12 și 30) trebuie sa aiba prioritate daca e ora.
	# Dar daca e 10 : 2 = 5? E greu de distins fara context.
	# Presupunem ca "si" e mai sigur pentru general (ora/scor), iar "impartit la" e specific.
	# Vom folosi regula anterioara cu "și" pentru : intre cifre, deoarece e mai comun in texte generale.
	# Daca doriti strict matematica, schimbati aici.
	# text = re.sub(r'(?<=\d):(?=\d)', ' împărțit la ', text) 
	# Pastram regula existenta: 12:30 -> 12 și 30
	text = re.sub(r'(\d):(\d)', r'\1 și \2', text)
	# Orice alt : devine virgula
	text = text.replace(':', ', ')

	# 3.6 Înmulțire
	text = text.replace("×", " ori ")
	text = text.replace("·", " ori ")
	# * devine ori doar dacă nu a fost prins la N* (adica e între spații sau cifre)
	text = text.replace("*", " ori ") 

	# 3.7 Simboluri matematice diverse
	math_mappings = {
		"∅": " mulțimea vidă ",
		"∞": " infinit ",
		"±": " plus minus ",
		"≤": " mai mic sau egal cu ",
		"<=": " mai mic sau egal cu ",
		"≥": " mai mare sau egal cu ",
		">=": " mai mare sau egal cu ",
		"≠": " diferit de ",
		"≈": " aproximativ egal cu ",
		"√": " radical din ",
		"∈": " aparține ",
		"∉": " nu aparține ",
		"⊂": " inclus în ",
		"⊃": " include ",
		"∪": " reunit cu ",
		"∩": " intersectat cu ",
		"∀": " oricare ar fi ",
		"∃": " există ",
		"∑": " suma ",
		"⇒": " rezultă că ",
		"⇔": " echivalent cu ",
		"=": " egal ", 
		"<": " mai mic ca ",
		">": " mai mare ca ",
		"%": " la sută ",
		"‰": " la mie "
	}
	for symbol, replacement in math_mappings.items():
		text = text.replace(symbol, replacement)

	# 3.8 Unități de măsură simple (doar dacă sunt precedate de numere)
	units = {
		"kg": "kilograme",
		"g": "grame",
		"mg": "miligrame",
		"km": "kilometri",
		"m": "metri", 
		"cm": "centimetri",
		"mm": "milimetri",
		"l": "litri",
		"ml": "mililitri",
		"V": "volți",
		"W": "wați",
		"Hz": "herți",
		"dB": "decibeli",
		"A": "amperi"
	}
	for unit, expansion in units.items():
		# Căutăm număr + spațiu opțional + unitate + word boundary
		pattern = r'(\d+)\s*' + unit + r'\b'
		text = re.sub(pattern, r'\1 ' + expansion, text)

	# 3.9 Punctuație și Separatori (Paranteze, acolade, slash)
	# Eliminăm parantezele neînchise corect transformându-le pe toate în virgule
	text = re.sub(r'[(){}\[\]|]', ', ', text)
	
	# Slash-ul rămas (care nu e km/h)
	text = text.replace("/", " supra ") 
	text = text.replace("\\", " supra ")

	# --- FINALIZARE ---

	# 4. Inlocuire linii de pauza (en-dash, em-dash) cu virgula
	# Atentie: nu inlocuim cratima (-) care leaga cuvinte (s-a, vis-à-vis)
	text = text.replace('–', ', ').replace('—', ', ')

	# 5. Eliminare ghilimele si apostroafe
	chars_to_remove = ['”', '“', '„', '»', '«', '"', "'", "’"]
	for c in chars_to_remove:
		text = text.replace(c, '')

	# 6. Gestionare initiale (J. M. Ziman -> J M Ziman)
	text = re.sub(r'\b([A-Z])\.', r'\1 ', text)

	# 7. Expandare cifre romane (ex: XX-lea)
	def replace_roman(match):
		roman = match.group(1)
		val = roman_to_int(roman)
		if val == 0: return match.group(0)
		try:
			words = num2words(val, lang='ro', to='ordinal')
			return words
		except:
			return match.group(0)

	text = re.sub(r'\b([IVXLCDM]+)-(lea|a)\b', replace_roman, text)
	text = text.replace('al al ', 'al ')
	text = text.replace('a a ', 'a ')

	# 8. Expandare numere folosind num2words
	def replace_num(match):
		num_str = match.group(0)
		try:
			clean_num = num_str.replace(',', '.')
			if '.' in clean_num:
				val = float(clean_num)
			else:
				val = int(clean_num)
			return num2words(val, lang='ro')
		except:
			return num_str

	text = re.sub(r'\b\d+([.,]\d+)?\b', replace_num, text)
	
	# 9. Curatenie finala punctuatie
	text = re.sub(r'\s+', ' ', text)       # spatii multiple -> un spatiu
	text = re.sub(r'\s+,', ',', text)      # spatiu inainte de virgula -> virgula
	text = re.sub(r',+', ',', text)        # virgule multiple -> o virgula
	text = re.sub(r',\s*,', ', ', text)    # virgula spatiu virgula -> virgula spatiu
	text = re.sub(r',\s*\.', '.', text)    # virgula punct -> punct
	text = text.strip(' ,')                # curatare capete
	
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
