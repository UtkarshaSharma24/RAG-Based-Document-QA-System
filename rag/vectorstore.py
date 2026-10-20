"""Vector store: ChromaDB (persistent) ya numpy (in-memory fallback)."""
import numpy as np


class NumpyStore:
    name = "numpy"

    def __init__(self):
        self.vecs, self.meta = None, []

    def add(self, vecs, metas):
        self.vecs, self.meta = vecs, metas

    def search(self, qvec, k):
        sims = self.vecs @ qvec          # vectors normalised hain => dot product = cosine
        idx = np.argsort(-sims)[:k]
        return [(self.meta[i], float(sims[i])) for i in idx]


class ChromaStore:
    """Har RAGPipeline ka apna collection (unique naam), taaki ek index doosre ko delete na kare."""
    name = "chroma"

    def __init__(self):
        import uuid
        import chromadb
        self.client = chromadb.EphemeralClient()       # index har baar naya banta hai, disk par rakhne ki zaroorat nahi
        self.col = self.client.create_collection(f"rag_{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"})

    def add(self, vecs, metas):
        self.col.add(ids=[m["chunk_id"] for m in metas], embeddings=vecs.tolist(),
                     documents=[m["text"] for m in metas],
                     metadatas=[{k: v for k, v in m.items() if k != "text"} for m in metas])

    def search(self, qvec, k):
        k = min(k, self.col.count())                    # chhote document mein k se kam chunks ho sakte hain
        r = self.col.query(query_embeddings=[qvec.tolist()], n_results=k)
        out = []
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            out.append(({**meta, "text": doc}, 1.0 - float(dist)))   # cosine distance -> similarity
        return out


def get_store(embedder_name):
    if embedder_name == "hf":        # TF-IDF ke dimensions corpus pe depend karte hain, isliye Chroma sirf HF ke saath
        try:
            import chromadb  # noqa: F401
            return ChromaStore()
        except ImportError:
            pass
    return NumpyStore()
