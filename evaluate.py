"""RAG ko test karta hai: Hit@1, Hit@3, MRR, answer accuracy, out-of-scope refusal, latency.
Run:  EMBEDDER=tfidf python evaluate.py     (ya EMBEDDER=hf agar sentence-transformers installed hai)"""
import json
import numpy as np
from rag import config
from rag.pipeline import RAGPipeline, NO_ANSWER

rag = RAGPipeline()
qs = json.load(open(config.ROOT / "data" / "eval_questions.json"))
inscope = [q for q in qs if q["in_scope"]]
oos = [q for q in qs if not q["in_scope"]]

hit1 = hit3 = ans_ok = 0
rr, lat = [], []
rows = []
for q in inscope:
    res = rag.ask(q["question"], k=3)
    docs = [s["doc_id"] for s in res["sources"]]
    h1 = docs[0] == q["doc_id"]
    h3 = q["doc_id"] in docs
    rank = docs.index(q["doc_id"]) + 1 if h3 else None
    kw_ok = q["keyword"].lower() in res["answer"].lower()
    hit1 += h1; hit3 += h3; ans_ok += kw_ok
    rr.append(1 / rank if rank else 0); lat.append(res["latency_ms"])
    rows.append({"q": q["question"], "expected": q["doc_id"], "retrieved": docs, "keyword_in_answer": kw_ok})

refused = sum(rag.ask(q["question"])["answer"] == NO_ANSWER for q in oos)
n = len(inscope)
result = {
    "embedder": rag.embedder.name, "vector_store": rag.store.name, "llm": config.LLM_BACKEND,
    "documents": len(rag.docs), "chunks": len(rag.chunks),
    "in_scope_questions": n, "out_of_scope_questions": len(oos),
    "hit_at_1": round(hit1 / n, 3), "hit_at_3": round(hit3 / n, 3), "mrr": round(float(np.mean(rr)), 3),
    "answer_keyword_accuracy": round(ans_ok / n, 3),
    "out_of_scope_refusal_rate": round(refused / len(oos), 3),
    "avg_latency_ms": round(float(np.mean(lat)), 2), "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
}
print(json.dumps(result, indent=2))
for r in rows:
    if not r["keyword_in_answer"] or r["expected"] != r["retrieved"][0]:
        print("MISS:", r)
(config.ROOT / "reports").mkdir(exist_ok=True)
json.dump({"summary": result, "details": rows}, open(config.ROOT / "reports" / f"eval_results_{rag.embedder.name}.json", "w"), indent=1)
