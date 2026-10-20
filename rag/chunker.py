"""Text ko chhote chunks mein todta hai (overlap ke saath)."""
import re
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def _simple_split(text, size, overlap):
    # Fallback: sentences ko jodkar chunk banata hai
    sents = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + " " + s if overlap else s
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)
    except ImportError:
        return _simple_split(text, size, overlap)


def make_chunks(docs):
    """Har chunk ko unique chunk_id (CHK00001...) aur source info deta hai."""
    out, n = [], 0
    for d in docs:
        for piece in split_text(d["text"]):
            n += 1
            out.append({"chunk_id": f"CHK{n:05d}", "doc_id": d["doc_id"], "source": d["source"],
                        "doc_type": d["doc_type"], "word_count": len(piece.split()), "text": piece})
    return out
