"""Saari settings ek jagah. Environment variable se override kar sakte ho."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = ROOT / "chroma_db"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))
TOP_K = int(os.getenv("TOP_K", 3))

# EMBEDDER: "hf" (sentence-transformers) ya "tfidf" (offline fallback)
EMBEDDER = os.getenv("EMBEDDER", "hf")
HF_MODEL = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM_BACKEND: "extractive" (no API needed), "openai", "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "extractive")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Agar best chunk ka score isse kam hai to system jawab nahi deta (hallucination se bachne ke liye)
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.12 if EMBEDDER == "tfidf" else 0.30))
