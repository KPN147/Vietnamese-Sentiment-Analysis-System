"""
Configuration file for sentiment analysis application
"""

# ===== MODEL CONFIGURATION =====

# PhoBERT fine-tuned model
SENTIMENT_MODEL_NAME = "KPN14/phobert_sentiment" 

# PhoWhisper model size
# Options: "vinai/PhoWhisper-large", "vinai/PhoWhisper-medium", "vinai/PhoWhisper-small"
SPEECH_MODEL_NAME = "vinai/PhoWhisper-large"

# ===== LABEL CONFIGURATION =====
# Adjust if your model has different label mapping
# True if: 0=Negative, 1=Positive
# False if: 0=Positive, 1=Negative
LABEL_0_IS_NEGATIVE = True

# Label display names
POSITIVE_LABEL = "Positive ✅"
NEGATIVE_LABEL = "Negative ❌"

# ===== PROCESSING CONFIGURATION =====
# Maximum text length (tokens)
MAX_TEXT_LENGTH = 256

# ===== SERVER CONFIGURATION =====
# Port to run the application
SERVER_PORT = 7860

# Create public link (True/False)
SHARE_PUBLICLY = False

# Server name (0.0.0.0 to allow access from local network)
SERVER_NAME = "0.0.0.0"
