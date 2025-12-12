# Configurare pentru Modelul Vision (Ollama)

# Modelul recomandat: "llava" sau "llama3.2-vision"
# Asigura-te ca ai rulat: "ollama pull llava" in terminal
MODEL_NAME = "qwen3-vl:235b-cloud" 

import os


def _load_prompt_file(filename: str, fallback: str) -> str:
	"""Încarcă promptul din prompts/ pentru editare ușoară, cu fallback la string."""
	base = os.path.dirname(os.path.realpath(__file__))
	path = os.path.join(base, "prompts", filename)
	try:
		with open(path, "r", encoding="utf-8") as f:
			content = f.read().strip()
		return content if content else fallback
	except Exception:
		return fallback


# Prompt pentru modul TTS (Text-to-Speech)
SYSTEM_PROMPT_TTS = _load_prompt_file(
	"system_tts.txt",
	"""
Ești un expert în transcrierea și normalizarea textului din manuale de matematică și fizică pentru limba română.

Returnează STRICT CSV cu separator | în format: TextBrut|TextNormalizat.
NU include ID-uri.
""".strip(),
)

# Prompt pentru modul STT (Speech-to-Text / Whisper)
SYSTEM_PROMPT_STT = _load_prompt_file(
	"system_stt.txt",
	"""
Ești un expert în transcrierea textului pentru Whisper.

Returnează STRICT CSV cu separator | în format: TextBrut|TextNormalizat.
NU include ID-uri.
""".strip(),
)
