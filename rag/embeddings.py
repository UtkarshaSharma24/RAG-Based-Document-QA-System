"""Embeddings: HuggingFace (best) ya TF-IDF (offline fallback)."""
import numpy as np
from .config import EMBEDDER, HF_MODEL


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def embed(self, texts):
        return self.vec.transform(texts).toarray().astype("float32")  # already L2-normalised


class HFEmbedder:
    name = "hf"

    def __init__(self, model=HF_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def fit(self, texts):
        return self

    def embed(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype="float32")


def get_embedder():
    if EMBEDDER == "hf":
        try:
            return HFEmbedder()
        except Exception as e:  # package ya model nahi mila
            print(f"[warn] HF embedder load nahi hua ({e}); TF-IDF use kar raha hoon")
    return TfidfEmbedder()
