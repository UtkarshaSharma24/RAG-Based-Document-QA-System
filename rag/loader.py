"""PDF / Word / TXT / Markdown files padhta hai."""
import re
from pathlib import Path


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")


def load_documents(folder: Path):
    """Har file ko {doc_id, source, doc_type, text} dict bana deta hai."""
    docs = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() not in (".txt", ".md", ".pdf", ".docx"):
            continue
        m = re.match(r"(DOC\d+)", path.name)
        doc_id = m.group(1) if m else f"DOC{len(docs) + 1:03d}"
        docs.append({"doc_id": doc_id, "source": path.name,
                     "doc_type": path.suffix.lstrip(".").upper(), "text": read_file(path)})
    return docs
