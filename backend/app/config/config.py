import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Hugging Face Token (optional, if you use HuggingFace for embeddings)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("✅ HF_TOKEN loaded successfully.")
else:
    print("⚠️ HF_TOKEN not found in environment.")

# Local Gemma 3 path (update to your actual path)
LOCAL_GEMMA_PATH = Path.home() / "Downloads" / "medical_chatbot" / "gemma_models" / "gemma-3-1b-it"

# Vector store and data configuration
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "/Users/nuzkhan/Downloads/medical chatbot/data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

