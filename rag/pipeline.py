"""End-to-end RAG pipeline: load -> chunk -> embed -> store -> retrieve -> answer."""
import time
from . import config
from .loader import load_documents
from .chunker import make_chunks
from .embeddings import get_embedder
from .vectorstore import get_store
from .llm import generate

NO_ANSWER = "I could not find this in the documents."
DISCLAIMER = ("RAG systems can make mistakes. Always check the source chunk. "
              "Do not use for legal or medical advice without expert review.")


class RAGPipeline:
    def __init__(self, folder=config.CORPUS_DIR):
        self.docs = load_documents(folder)
        self.chunks = make_chunks(self.docs)
        self.embedder = get_embedder().fit([c["text"] for c in self.chunks])
        self.store = get_store(self.embedder.name)
        self.store.add(self.embedder.embed([c["text"] for c in self.chunks]), self.chunks)

    def retrieve(self, question, k=config.TOP_K):
        return self.store.search(self.embedder.embed([question])[0], k)

    def ask(self, question, k=config.TOP_K):
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        sources = [{**m, "retrieval_score": round(s, 4)} for m, s in hits]
        if not hits or hits[0][1] < config.MIN_SCORE:
            answer = NO_ANSWER
        else:
            answer = generate(question, [m["text"] for m, _ in hits],
                              embed=self.embedder.embed if self.embedder.name == "hf" else None)
        return {"question": question, "answer": answer, "sources": sources,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
