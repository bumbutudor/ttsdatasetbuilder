# Configurare pentru Modelul Vision (Ollama sau OpenAI)
import os

# Selectează providerul: "ollama" sau "openai"
AI_PROVIDER = "ollama"

# Configurare Ollama
OLLAMA_MODEL_NAME = "qwen3-vl:235b-cloud"

# Configurare OpenAI
OPENAI_MODEL_NAME = "gpt-5-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Păstrăm MODEL_NAME pentru compatibilitate (va fi setat dinamic în main sau folosit cel de Ollama ca default)
MODEL_NAME = OLLAMA_MODEL_NAME if AI_PROVIDER == "ollama" else OPENAI_MODEL_NAME


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
