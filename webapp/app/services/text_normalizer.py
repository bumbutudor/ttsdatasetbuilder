"""Text normalization services for TTS and STT."""
import re
from typing import Literal

try:
    from num2words import num2words
except ImportError:
    num2words = None


# Romanian abbreviation mappings for TTS
TTS_MAPPINGS = {
    # Symbols
    '€': 'euro',
    '$': 'dolari',
    '%': 'la sută',
    '+': 'plus',
    '=': 'egal',
    '&': 'și',
    '"': '"',
    '„': '"',
    '»': '"',
    '«': '"',
    
    # Titles
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
    
    # Addresses
    'str.': 'strada',
    'b-dul': 'bulevardul',
    'bd.': 'bulevardul',
    'nr.': 'numărul',
    'bl.': 'blocul',
    'sc.': 'scara',
    'et.': 'etajul',
    'ap.': 'apartamentul',
    'sect.': 'sectorul',
    'jud.': 'județul',
    
    # Common expressions
    'etc.': 'și așa mai departe',
    'ș.a.': 'și altele',
    'ex.': 'exemplu',
    'cca.': 'circa',
    'aprox.': 'aproximativ',
    'pt.': 'pentru',
    'ptr.': 'pentru',
    'pag.': 'pagina',
    'vol.': 'volumul',
    'art.': 'articolul',
    'cap.': 'capitolul',
    
    # Time
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
    'î.Hr.': 'înainte de Hristos',
    'd.Hr.': 'după Hristos',
    
    # Acronyms - read as words
    'NATO': 'nato',
    'UNESCO': 'unesco',
    'ONU': 'onu',
    'NASA': 'nasa',
    
    # Acronyms - spelled out
    'SUA': 'S U A',
    'UE': 'U E',
    'TV': 'te ve',
    'OK': 'ochei',
}


def roman_to_int(s: str) -> int:
    """Convert Roman numeral to integer."""
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


def normalize_tts(text: str) -> str:
    """Normalize text for TTS: expand abbreviations, numbers to words."""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove square bracket content (citations)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Expand abbreviations
    for k, v in TTS_MAPPINGS.items():
        if k.endswith('.'):
            k_esc = re.escape(k)
            text = re.sub(r'(?<!\w)' + k_esc + r'(?!\w)', v, text)
        else:
            text = text.replace(k, v)
    
    # Temperature and units
    text = text.replace("°C", " grade Celsius")
    text = text.replace("°F", " grade Fahrenheit")
    text = text.replace("°", " grade")
    text = text.replace("km/h", " kilometri pe oră")
    text = text.replace("m/s", " metri pe secundă")
    
    # Math symbols
    text = re.sub(r'(?<=\d)-(?=\d)', ' minus ', text)
    text = text.replace("−", " minus ")
    text = text.replace("×", " ori ")
    text = text.replace("·", " ori ")
    text = re.sub(r'(\d):(\d)', r'\1 și \2', text)
    text = text.replace(':', ', ')
    
    # Dashes to commas
    text = text.replace('–', ', ').replace('—', ', ')
    
    # Remove quotes
    for c in ['"', '"', '„', '»', '«', '"', "'", "'"]:
        text = text.replace(c, '')
    
    # Handle initials
    text = re.sub(r'\b([A-Z])\.', r'\1 ', text)
    
    # Expand Roman numerals with ordinal suffix
    def replace_roman(match):
        roman = match.group(1)
        val = roman_to_int(roman)
        if val == 0:
            return match.group(0)
        if num2words:
            try:
                return num2words(val, lang='ro', to='ordinal')
            except:
                pass
        return match.group(0)
    
    text = re.sub(r'\b([IVXLCDM]+)-(lea|a)\b', replace_roman, text)
    
    # Expand numbers to words
    if num2words:
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
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',+', ',', text)
    text = re.sub(r',\s*\.', '.', text)
    text = text.strip(' ,')
    
    return text


def normalize_stt(text: str) -> str:
    """Normalize text for STT: keep numbers, fix spacing and punctuation."""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove square bracket content (citations)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Replace parentheses with commas
    text = text.replace('(', ', ').replace(')', ', ')
    
    # Standardize dashes
    text = re.sub(r'\s*[–—]\s*', ' – ', text)
    
    # Standardize quotes to Romanian format
    text = text.replace('«', '„').replace('»', '"')
    text = text.replace('"', '„')
    text = re.sub(r'(^|\s)"', r'\1„', text)
    text = text.replace('"', '"')
    
    # Fix spacing and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,?!:;])', r'\1', text)
    text = re.sub(r'([.,?!:;])(?=[a-zA-Z])', r'\1 ', text)
    
    # Clean double commas
    text = re.sub(r',\s*,', ', ', text)
    text = re.sub(r',\s*\.', '.', text)
    
    return text.strip()


def normalize_text(text: str, mode: Literal["TTS", "STT"]) -> str:
    """Normalize text based on mode."""
    if mode == "TTS":
        return normalize_tts(text)
    else:
        return normalize_stt(text)

