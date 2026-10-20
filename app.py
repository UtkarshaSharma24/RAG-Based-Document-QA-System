"""Streamlit UI  ->  streamlit run app.py"""
import hashlib
import html
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rag import config
from rag.llm import STOP
from rag.pipeline import DISCLAIMER, NO_ANSWER, RAGPipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1150px; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }

.hero { background: linear-gradient(120deg, #4338CA 0%, #6D28D9 55%, #9333EA 100%); border-radius: 20px;
        padding: 34px 38px; color: #fff; box-shadow: 0 10px 30px rgba(79,70,229,.25); margin-bottom: 18px; }
.hero .eyebrow { font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; opacity: .8; font-weight: 600; margin-bottom: 8px; }
.hero h1 { margin: 0 0 8px 0; padding: 0; font-size: 2.15rem; font-weight: 700; color: #fff; letter-spacing: -.5px; }
.hero p { margin: 0; font-size: 1.02rem; color: rgba(255,255,255,.9); max-width: 740px; line-height: 1.6; }
.badges { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
.badge { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3); padding: 4px 12px;
         border-radius: 999px; font-size: .78rem; font-weight: 500; color: #fff; }

.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 14px 0; }
.stat { background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.stat .v { font-size: 1.65rem; font-weight: 700; color: #111827; line-height: 1.1; }
.stat .l { font-size: .72rem; color: #6B7280; margin-top: 6px; text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }

.section-title { font-size: .76rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #6B7280; margin: 22px 0 10px 0; }

.answer-card { background: #fff; border: 1px solid #E5E7EB; border-left: 5px solid #4F46E5; border-radius: 14px;
               padding: 22px 26px; box-shadow: 0 4px 14px rgba(16,24,40,.06); margin-top: 6px; }
.answer-card.none { border-left-color: #DC2626; background: #FEF2F2; }
.answer-q { font-size: .83rem; color: #6B7280; margin-bottom: 10px; }
.answer-q b { color: #374151; font-weight: 600; }
.answer-text { font-size: 1.12rem; line-height: 1.7; color: #111827; }
.answer-hint { font-size: .88rem; color: #7F1D1D; margin-top: 8px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.pill { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .74rem; font-weight: 600; }
.pill.strong { background: #DCFCE7; color: #166534; }
.pill.moderate { background: #FEF3C7; color: #92400E; }
.pill.none { background: #FEE2E2; color: #991B1B; }
.pill.info { background: #EEF2FF; color: #3730A3; }
.pill.grey { background: #F3F4F6; color: #374151; }

.src { background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 16px 20px; margin-bottom: 12px; }
.src-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.rank { width: 26px; height: 26px; border-radius: 8px; background: #4F46E5; color: #fff; font-size: .8rem; font-weight: 700;
        display: inline-flex; align-items: center; justify-content: center; }
.src-file { font-size: .8rem; color: #6B7280; }
.score { margin-left: auto; font-size: .8rem; font-weight: 600; color: #374151; }
.bar { height: 6px; background: #EEF2FF; border-radius: 999px; overflow: hidden; margin-bottom: 12px; }
.bar > span { display: block; height: 100%; background: linear-gradient(90deg, #6366F1, #9333EA); border-radius: 999px; }
.src-text { font-size: .93rem; line-height: 1.7; color: #374151; }
.src-text mark { background: #FEF08A; color: #111827; padding: 0 2px; border-radius: 3px; }

.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.step { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 16px; font-size: .86rem; color: #374151; line-height: 1.5; }
.step b { display: block; color: #4F46E5; margin-bottom: 4px; }
.docitem { font-size: .83rem; color: #374151; padding: 6px 0; border-bottom: 1px dashed #E5E7EB; }
.docitem span { color: #6B7280; font-size: .75rem; }
.disclaimer { background: #FFFBEB; border: 1px solid #FDE68A; color: #78350F; border-radius: 12px; padding: 12px 16px;
              font-size: .85rem; margin-top: 28px; }
.docview { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 12px; padding: 16px 20px; max-height: 340px;
           overflow-y: auto; font-size: .9rem; line-height: 1.7; color: #374151; white-space: pre-wrap; }

.stButton > button { width: 100%; border-radius: 10px; font-weight: 500; font-size: .84rem; text-align: left;
                     min-height: 3rem; border: 1px solid #E5E7EB; background: #fff; color: #374151; }
.stButton > button:hover { border-color: #6366F1; color: #4338CA; background: #EEF2FF; }
[data-testid="stFormSubmitButton"] > button { width: 100%; border-radius: 10px; font-weight: 600; min-height: 2.6rem; }
@media (max-width: 800px) { .stats, .steps { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 1.6rem; } }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

EXAMPLES = [
    "How much money does PM-KISAN give per year?",
    "What is the health cover per family under Ayushman Bharat PM-JAY?",
    "How many days does the PIO have to reply to an RTI application?",
    "Which articles cover the Right to Equality?",
    "What is the overdraft limit for Jan Dhan accounts?",
    "How do I create a virtual environment in Python?",
]
EMBEDDER_LABEL = {"hf": "MiniLM-L6-v2 embeddings", "tfidf": "TF-IDF vectors"}
STORE_LABEL = {"chroma": "ChromaDB vector store", "numpy": "In-memory vector index"}
LLM_LABEL = {"extractive": "Extractive answers", "openai": "OpenAI answers", "ollama": "Ollama (local LLM)"}


@st.cache_resource(show_spinner="Indexing documents...")
def build(folder: str) -> RAGPipeline:
    return RAGPipeline(Path(folder))


def save_uploads(files) -> str:
    """Same files -> same folder, so the index is cached and not rebuilt on every click."""
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(f.getvalue())
    folder = Path(tempfile.gettempdir()) / f"rag_upload_{h.hexdigest()[:12]}"
    if not folder.exists():
        folder.mkdir(parents=True)
        for f in files:
            (folder / Path(f.name).name).write_bytes(f.getvalue())
    return str(folder)


def highlight(text: str, question: str) -> str:
    terms = {w for w in re.findall(r"[a-z0-9]+", question.lower())
             if len(w) > 3 and w not in STOP and w not in {"amp", "quot", "x27"}}
    safe = html.escape(text)
    if not terms:
        return safe
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b", re.I)
    return pattern.sub(r"<mark>\1</mark>", safe)


def use_example(q: str):
    st.session_state["q_input"] = q
    st.session_state["query"] = q


st.session_state.setdefault("query", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_logged", None)

# ---------------------------------------------------------------- sidebar (inputs)
with st.sidebar:
    st.markdown("### 📄 RAG Q&A")
    st.markdown("<div class='section-title'>Knowledge base</div>", unsafe_allow_html=True)
    files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "docx", "txt", "md"],
                             label_visibility="collapsed")
    st.caption("PDF, DOCX, TXT or MD. Leave empty to use the built-in sample library.")
    st.markdown("<div class='section-title'>Retrieval</div>", unsafe_allow_html=True)
    k = st.slider("Chunks retrieved (Top-K)", 1, 8, config.TOP_K,
                  help="How many of the most similar chunks are used to build the answer.")

using_samples = not files
try:
    rag = build(str(config.CORPUS_DIR) if using_samples else save_uploads(files))
    if not rag.chunks:
        raise ValueError("No readable text found. Scanned PDFs need OCR first.")
except Exception as e:  # unreadable file, empty text, missing package...
    st.error(f"Could not index the documents: {e}")
    st.stop()

# ---------------------------------------------------------------- sidebar (info)
with st.sidebar:
    st.markdown("<div class='section-title'>Indexed documents</div>", unsafe_allow_html=True)
    st.markdown("".join(f"<div class='docitem'><b>{html.escape(d['doc_id'])}</b> &nbsp;{html.escape(d['source'][:34])}</div>"
                        for d in rag.docs), unsafe_allow_html=True)
    if st.session_state["history"]:
        st.markdown("<div class='section-title'>Recent questions</div>", unsafe_allow_html=True)
        for h in st.session_state["history"][-6:][::-1]:
            st.markdown(f"<div class='docitem'>{html.escape(h['q'])}<br><span>{html.escape(h['a'][:70])}...</span></div>",
                        unsafe_allow_html=True)
    st.caption(f"Embedder: {config.EMBEDDER} · LLM: {config.LLM_BACKEND}")

# ---------------------------------------------------------------- hero + stats
avg_words = round(sum(c["word_count"] for c in rag.chunks) / len(rag.chunks))
badges = [EMBEDDER_LABEL.get(rag.embedder.name, rag.embedder.name), STORE_LABEL.get(rag.store.name, rag.store.name),
          LLM_LABEL.get(config.LLM_BACKEND, config.LLM_BACKEND), "Source shown with every answer"]
st.markdown(
    "<div class='hero'><div class='eyebrow'>Retrieval-Augmented Generation</div>"
    "<h1>Ask your documents anything</h1>"
    "<p>Type a question in plain English. The system finds the most relevant passages in your documents, "
    "answers only from them, and shows the exact source next to every answer.</p>"
    "<div class='badges'>" + "".join(f"<span class='badge'>{html.escape(b)}</span>" for b in badges) + "</div></div>",
    unsafe_allow_html=True)
stats = [(len(rag.docs), "Documents"), (len(rag.chunks), "Chunks indexed"), (k, "Chunks per answer"), (avg_words, "Avg. words / chunk")]
st.markdown("<div class='stats'>" + "".join(f"<div class='stat'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
                                            for v, l in stats) + "</div>", unsafe_allow_html=True)

tab_ask, tab_kb = st.tabs(["💬  Ask a question", "📚  Knowledge base"])

# ---------------------------------------------------------------- ask tab
with tab_ask:
    if using_samples:
        st.markdown("<div class='section-title'>Try an example</div>", unsafe_allow_html=True)
        for row in (EXAMPLES[:3], EXAMPLES[3:]):
            cols = st.columns(3)
            for col, q in zip(cols, row):
                with col:
                    st.button(q, key=f"ex_{q}", on_click=use_example, args=(q,))

    st.markdown("<div class='section-title'>Your question</div>", unsafe_allow_html=True)
    with st.form("ask_form", clear_on_submit=False):
        c_in, c_btn = st.columns([5, 1])
        c_in.text_input("Your question", key="q_input", label_visibility="collapsed",
                        placeholder="e.g. How much money does PM-KISAN give per year?")
        submitted = c_btn.form_submit_button("Ask", type="primary")
    if submitted:
        st.session_state["query"] = st.session_state["q_input"].strip()

    query = st.session_state["query"]
    if query:
        try:
            with st.spinner("Searching your documents..."):
                res = rag.ask(query, k)
        except Exception as e:   # stale index etc.: clear cache so the next click rebuilds it
            st.cache_resource.clear()
            st.error(f"Something went wrong while searching ({e}). Please press Ask again.")
            st.stop()
        sources = res["sources"]
        refused = res["answer"] == NO_ANSWER
        top = sources[0]["retrieval_score"] if sources else 0.0
        if refused:
            conf, conf_label = "none", "No confident answer"
        elif top >= 1.6 * config.MIN_SCORE:
            conf, conf_label = "strong", "Strong match"
        else:
            conf, conf_label = "moderate", "Moderate match"

        key = (query, k, len(rag.chunks))
        if st.session_state["last_logged"] != key:
            st.session_state["last_logged"] = key
            st.session_state["history"].append({"q": query, "a": res["answer"]})

        st.markdown("<div class='section-title'>Answer</div>", unsafe_allow_html=True)
        body = ("<div class='answer-text'>I could not find this in the documents.</div>"
                "<div class='answer-hint'>Try rephrasing the question, or check the closest passages below.</div>"
                if refused else f"<div class='answer-text'>{html.escape(res['answer'])}</div>")
        best = f"{sources[0]['doc_id']} · {top:.2f}" if sources else "-"
        st.markdown(
            f"<div class='answer-card{' none' if refused else ''}'>"
            f"<div class='answer-q'>Question &nbsp;·&nbsp; <b>{html.escape(query)}</b></div>{body}"
            f"<div class='meta'><span class='pill {conf}'>{conf_label}</span>"
            f"<span class='pill info'>⏱ {res['latency_ms']} ms</span>"
            f"<span class='pill grey'>{len(sources)} chunks used</span>"
            f"<span class='pill grey'>Best match: {html.escape(best)}</span></div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>" + ("Closest passages (below confidence threshold)" if refused
                                                      else "Sources used for this answer") + "</div>", unsafe_allow_html=True)
        for i, s in enumerate(sources, 1):
            width = max(2, min(100, round(s["retrieval_score"] * 100)))
            st.markdown(
                f"<div class='src'><div class='src-head'><span class='rank'>{i}</span>"
                f"<span class='pill info'>{html.escape(s['doc_id'])}</span>"
                f"<span class='pill grey'>{html.escape(s['chunk_id'])}</span>"
                f"<span class='src-file'>{html.escape(s['source'])}</span>"
                f"<span class='score'>Similarity {s['retrieval_score']:.3f}</span></div>"
                f"<div class='bar'><span style='width:{width}%'></span></div>"
                f"<div class='src-text'>{highlight(s['text'], query)}</div></div>", unsafe_allow_html=True)

    with st.expander("How it works"):
        st.markdown(
            "<div class='steps'>"
            "<div class='step'><b>1 · Chunk</b>Documents are split into small overlapping pieces of text.</div>"
            "<div class='step'><b>2 · Embed</b>Each chunk is turned into a vector that captures its meaning.</div>"
            "<div class='step'><b>3 · Retrieve</b>Your question is embedded and the most similar chunks are found.</div>"
            "<div class='step'><b>4 · Answer</b>The answer is built only from those chunks, and the sources are shown.</div>"
            "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- knowledge base tab
with tab_kb:
    rows = []
    for d in rag.docs:
        ch = [c for c in rag.chunks if c["doc_id"] == d["doc_id"]]
        rows.append({"Doc ID": d["doc_id"], "File": d["source"], "Type": d["doc_type"],
                     "Chunks": len(ch), "Words": sum(c["word_count"] for c in ch)})
    df = pd.DataFrame(rows)
    st.markdown("<div class='section-title'>Documents in the index</div>", unsafe_allow_html=True)
    st.dataframe(df, hide_index=True)
    st.markdown("<div class='section-title'>Chunks per document</div>", unsafe_allow_html=True)
    st.bar_chart(df.set_index("Doc ID")["Chunks"])
    st.markdown("<div class='section-title'>Preview a document</div>", unsafe_allow_html=True)
    pick = st.selectbox("Document", [d["doc_id"] + "  ·  " + d["source"] for d in rag.docs], label_visibility="collapsed")
    doc = next(d for d in rag.docs if d["doc_id"] == pick.split("  ·  ")[0])
    st.markdown(f"<div class='docview'>{html.escape(doc['text'])}</div>", unsafe_allow_html=True)

st.markdown(f"<div class='disclaimer'>⚠️ <b>Please verify.</b> {html.escape(DISCLAIMER)}</div>", unsafe_allow_html=True)
