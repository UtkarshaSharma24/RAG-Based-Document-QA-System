"""Answer generation. Hamesha sirf diye gaye context se jawab dena hai."""
import re
from .config import LLM_BACKEND, OPENAI_MODEL, OLLAMA_MODEL

PROMPT = """You are a careful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "I could not find this in the documents."
Keep the answer short and factual.

Context:
{context}

Question: {question}
Answer:"""

STOP = set("a an the is are was were of to in on for and or by with what which who how when where does do did can i it be as at from that this".split())


def _tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def extractive_answer(question, contexts, n_sentences=3, embed=None):
    """LLM ke bina answer: retrieved text ke sentences ko question se similarity ke basis par rank karta hai.
    embed diya ho (HF model) to meaning se rank hota hai ("money" ~ "rupees"), warna TF-IDF se."""
    sents = []
    for c in contexts:
        for s in re.split(r"(?<=[.!?])\s+|\n+", c):
            s = s.strip()
            words = len(s.split())
            is_sentence = words >= 4 and s.endswith((".", "!", "?"))
            is_field = ":" in s and words >= 2          # "Symptoms: Chills, Chest pain" jaisi report/form lines
            if (is_sentence or is_field) and s not in sents:   # heading lines skip
                sents.append(s)
    if not sents:
        return ""
    if embed is not None:
        vecs = embed(sents + [question])
        sims = vecs[:-1] @ vecs[-1]
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(sents + [question])
        sims = (vec.transform(sents) @ vec.transform([question]).T).toarray().ravel()
    # Sawal ke type ke hisaab se sahi tarah ka number wale sentence ko boost karo
    q = question.lower()
    year = r"(1[89]|20)\d{2}"
    if re.search(r"\bwhen\b|\bwhat year\b|\bwhich year\b", q):            # date / year chahiye
        has = lambda t: bool(re.search(r"\b" + year + r"\b", t))
    elif re.search(r"\bhow (much|many|long|old)\b|\b(limit|fee|amount|cover|strength|maximum|minimum)\b", q):
        # quantity chahiye: saal (2019 jaise) ko chhodkar koi bhi number
        has = lambda t: any(not re.fullmatch(year, n) for n in re.findall(r"\d[\d,]*", t))
    else:
        has = None
    if has:
        sims = [float(x) + (0.15 if has(sents[i]) else 0.0) for i, x in enumerate(sims)]
    top = sorted(sorted(range(len(sents)), key=lambda i: -sims[i])[:n_sentences])  # original order mein
    return " ".join(sents[i] for i in top)


def generate(question, contexts, embed=None):
    if LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        return llm.invoke(PROMPT.format(context="\n\n".join(contexts), question=question)).content
    if LLM_BACKEND == "ollama":
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        return llm.invoke(PROMPT.format(context="\n\n".join(contexts), question=question)).content
    return extractive_answer(question, contexts, embed=embed)
