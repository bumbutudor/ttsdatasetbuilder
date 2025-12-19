# stt_config.py

# Configuration for Speech-to-Text Model

# Options: 'ggml' (local .bin file) or 'huggingface' (download from HF)
MODEL_SOURCE = 'huggingface' 
# MODEL_SOURCE = 'ggml'

# For 'ggml': filename in 'models' folder (e.g., 'ggml-whisper-medium-romanian.bin')
# For 'huggingface': Model ID (e.g., 'TransferRapid/whisper-large-v3-turbo_ro')
MODEL_NAME = 'iRaduS/whisper-romanian-finetune'
# MODEL_NAME = 'ggml-whisper-medium-romanian.bin'

# Available HuggingFace models for reference:
# - "TransferRapid/whisper-large-v3-turbo_ro"
# - "iRaduS/whisper-romanian-finetune"
# - "gigant/whisper-medium-romanian"
# - "readerbench/whisper-ro"
