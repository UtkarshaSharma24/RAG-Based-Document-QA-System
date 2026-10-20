# RAG-Based Document Q&A System
HCLTech Career Shaper Programme - III Year Industry Project (Project No. 72)

Documents (PDF / DOCX / TXT) ko chunks mein todkar vector DB mein store karta hai, sawal aane par relevant chunks dhoondhta hai
aur sirf unhi se answer deta hai. Answer ke saath source chunk aur disclaimer hamesha dikhta hai.

## Folder structure
```
app.py                 Streamlit UI
evaluate.py            Hit@K, MRR, answer accuracy, refusal rate, latency
eda.py                 Dataset cleaning + 5 visualisations
rag/                   loader, chunker, embeddings, vectorstore, llm, pipeline, config
data/corpus/           8 sample documents (own simplified summaries - official source se verify karo)
data/dataset.csv       5,000 rows (500 provided + 4,500 generated)
data/data_dictionary.csv
data/eval_questions.json   33 in-scope + 5 out-of-scope questions
reports/               figures, eda_summary.json, eval_results_*.json
```

## Run
```
python -m venv venv
venv\Scripts\activate            (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Settings (environment variables): `EMBEDDER=hf|tfidf`, `LLM_BACKEND=extractive|openai|ollama`, `TOP_K`, `MIN_SCORE`,
`OPENAI_API_KEY` (agar openai use karna hai).

## Reproduce results
```
python eda.py
EMBEDDER=tfidf python evaluate.py     # offline baseline (jo report mein hai)
python evaluate.py                    # MiniLM + Chroma  -> report ki Table 2 ka dusra column bharo
```

## Important
- LangChain / ChromaDB / sentence-transformers / Streamlit wale paths build environment mein install nahi ho sake (no internet),
  isliye unhe apni machine par ek baar chala ke check karna. Offline TF-IDF fallback path poori tarah test hua hai.
- Not for legal or medical advice. Always check the source chunk.
