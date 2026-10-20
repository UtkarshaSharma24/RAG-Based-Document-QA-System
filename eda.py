"""EDA on dataset.csv: cleaning + 5 visualisations + summary json.   Run: python eda.py"""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# ---------- 1. Load + clean ----------
raw = pd.read_csv(ROOT / "data" / "dataset.csv")
df = raw.copy()
report = {"rows_raw": len(raw), "nulls": int(raw.isna().sum().sum()),
          "duplicate_rows": int(raw.duplicated().sum()), "duplicate_chunk_ids": int(raw.chunk_id.duplicated().sum())}
df = df.drop_duplicates().drop_duplicates("chunk_id")
for c in ["doc_id", "doc_type", "chunk_id", "query_type"]:
    df[c] = df[c].astype(str).str.strip()
df["doc_type"] = df["doc_type"].replace({"Markdown": "Markdown", "TXT": "TXT"})
valid = (df.word_count.between(1, 2000) & df.retrieval_score.between(0, 1) & df.answer_quality_score.between(0, 1))
report["out_of_range_rows"] = int((~valid).sum())
df = df[valid]
df["quality_band"] = pd.cut(df.answer_quality_score, [0, 0.7, 0.85, 1.0], labels=["low", "medium", "high"])
df["retrieval_band"] = pd.cut(df.retrieval_score, [0, 0.6, 0.8, 1.0], labels=["weak", "good", "strong"])
df.to_csv(ROOT / "data" / "dataset_clean.csv", index=False)
report["rows_clean"] = len(df)

# ---------- 2. Summary stats ----------
num = ["word_count", "retrieval_score", "answer_quality_score"]
report["describe"] = df[num].describe().round(3).to_dict()
report["doc_type_counts"] = df.doc_type.value_counts().to_dict()
report["query_type_counts"] = df.query_type.value_counts().to_dict()
report["docs"] = int(df.doc_id.nunique())
report["corr"] = df[num].corr().round(3).to_dict()
report["mean_by_query_type"] = df.groupby("query_type")[["retrieval_score", "answer_quality_score"]].mean().round(3).to_dict()
report["mean_by_doc_type"] = df.groupby("doc_type")[["retrieval_score", "answer_quality_score"]].mean().round(3).to_dict()
report["quality_band_share"] = df.quality_band.value_counts(normalize=True).round(3).to_dict()
report["retrieval_band_share"] = df.retrieval_band.value_counts(normalize=True).round(3).to_dict()
report["share_low_retrieval_lt_0_6"] = round(float((df.retrieval_score < 0.6).mean()), 3)

# ---------- 3. Visualisations ----------
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, c, col in zip(ax, num, ["#4C72B0", "#55A868", "#C44E52"]):
    sns.histplot(df[c], bins=30, ax=a, color=col); a.set_title(f"Distribution of {c}")
plt.tight_layout(); plt.savefig(FIG / "fig1_distributions.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.countplot(data=df, x="doc_type", order=df.doc_type.value_counts().index, ax=ax[0], color="#4C72B0"); ax[0].set_title("Chunks per document type")
sns.countplot(data=df, x="query_type", order=df.query_type.value_counts().index, ax=ax[1], color="#55A868"); ax[1].set_title("Chunks per query type")
ax[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(FIG / "fig2_counts.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.boxplot(data=df, x="query_type", y="answer_quality_score", ax=ax[0], color="#8172B2"); ax[0].set_title("Answer quality by query type"); ax[0].tick_params(axis="x", rotation=20)
sns.boxplot(data=df, x="doc_type", y="retrieval_score", ax=ax[1], color="#CCB974"); ax[1].set_title("Retrieval score by document type")
plt.tight_layout(); plt.savefig(FIG / "fig3_boxplots.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(df[num].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax[0]); ax[0].set_title("Correlation heatmap")
sns.scatterplot(data=df.sample(1500, random_state=1), x="retrieval_score", y="answer_quality_score", alpha=0.35, ax=ax[1])
ax[1].set_title("Retrieval score vs answer quality")
plt.tight_layout(); plt.savefig(FIG / "fig4_correlation.png", dpi=150); plt.close()

top = df.groupby("doc_id").answer_quality_score.mean().sort_values()
fig, ax = plt.subplots(figsize=(11, 3.8))
pd.concat([top.head(5), top.tail(5)]).plot.bar(ax=ax, color="#4C72B0")
ax.set_ylim(0.7, 0.9); ax.set_title("Lowest 5 and highest 5 documents by mean answer quality"); ax.set_ylabel("mean quality")
plt.tight_layout(); plt.savefig(FIG / "fig5_doc_quality.png", dpi=150); plt.close()

json.dump(report, open(ROOT / "reports" / "eda_summary.json", "w"), indent=1, default=str)
print(json.dumps({k: report[k] for k in ["rows_raw", "rows_clean", "nulls", "duplicate_rows", "out_of_range_rows", "docs", "corr", "quality_band_share", "share_low_retrieval_lt_0_6"]}, indent=1, default=str))
print(report["describe"]); print(report["mean_by_query_type"])
