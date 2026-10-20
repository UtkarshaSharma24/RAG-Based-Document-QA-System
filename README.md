# RAG-Based Document Q&A System

A Python-based document question-answering application that retrieves relevant information from a document collection and helps users find answers using Retrieval-Augmented Generation (RAG) concepts.

## Overview

Reading multiple documents to find specific information can be time-consuming. This project provides a simple way to ask questions about a collection of documents and retrieve relevant content.

The system loads documents, splits their text into chunks, retrieves relevant information based on a user's query, and uses the configured answer-generation method to produce a response.

## Features

- **Document Processing:** Supports TXT, Markdown, PDF, and DOCX files.
- **Text Chunking:** Splits documents into smaller sections for retrieval.
- **Information Retrieval:** Supports embedding-based retrieval and TF-IDF.
- **Question Answering:** Produces answers using retrieved document content.
- **Streamlit Interface:** Provides a simple, interactive user interface.
- **Evaluation:** Includes scripts for testing retrieval performance.
- **Data Analysis:** Includes an EDA script for exploring the project datasets.
- **Flexible Configuration:** Supports different retrieval and answer-generation configurations.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| User Interface | Streamlit |
| Machine Learning | Scikit-learn, NumPy, Pandas |
| Embeddings | Sentence Transformers, TF-IDF |
| Vector Store | ChromaDB |
| Document Processing | PyPDF, python-docx |
| LLM Integration | OpenAI / Ollama, depending on configuration |
| Visualization | Matplotlib, Seaborn |

## Project Structure

```text
RAG-Based_Document_QA_System/
│
├── app.py
├── requirements.txt
├── eda.py
├── evaluate.py
│
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── loader.py
│   ├── vectorstore.py
│   ├── chunker.py
│   └── pipeline.py
│
├── data/
│   ├── corpus/
│   ├── dataset.csv
│   ├── dataset_clean.csv
│   ├── eval_questions.json
│   └── data_dictionary.csv
│
├── figures/
├── reports/
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip package manager
- Git (optional)

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Based_Document_QA_System
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create a Virtual Environment

On Windows, open PowerShell in the project directory and run:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## Retrieval Configuration

The project supports configurable retrieval methods, depending on the implementation and settings in `rag/config.py`.

To use TF-IDF retrieval in Windows PowerShell:

```powershell
$env:EMBEDDER = "tfidf"
python -m streamlit run app.py
```

Embedding-based retrieval may require downloading a model the first time it runs.

## Evaluation

The repository includes scripts for exploratory data analysis and evaluating retrieval performance.

Run the evaluation script:

```bash
python evaluate.py
```

Run exploratory data analysis:

```bash
python eda.py
```

To evaluate using TF-IDF in PowerShell:

```powershell
$env:EMBEDDER = "tfidf"
python evaluate.py
```

Evaluation may include metrics such as:

- **Hit@1:** Whether a relevant result appears first.
- **Hit@3:** Whether a relevant result appears in the top three.
- **Mean Reciprocal Rank (MRR):** Measures the ranking position of the first relevant result.
- **Answer Accuracy:** Checks answers against predefined evaluation criteria.
- **Refusal Rate:** Tracks queries the system declines to answer.
- **Latency:** Measures query-processing time.

Actual results depend on the dataset, configuration, and evaluation run. Run the scripts before publishing specific performance figures.

## Optional LLM Configuration

An external LLM provider may be configured if supported by your selected backend.

For example, with an OpenAI-compatible configuration in PowerShell:

```powershell
$env:LLM_BACKEND = "openai"
$env:OPENAI_API_KEY = "YOUR_API_KEY_HERE"
python -m streamlit run app.py
```

Replace the placeholder with your own API key. Never commit API keys or other secrets to GitHub.

## Limitations

- Answer quality depends on the documents available to the system.
- Retrieval may return irrelevant results for ambiguous or unsupported questions.
- Embedding models may need an internet connection for initial setup.
- External LLM services may require an API key and can incur usage costs.
- Evaluation scores depend on the quality of the test dataset and evaluation criteria.

## Future Improvements

- Add more document formats.
- Improve chunking and retrieval strategies.
- Introduce reranking for more relevant results.
- Improve source citations and answer traceability.
- Expand the evaluation dataset.
- Add document management and user authentication.
- Deploy the application to a cloud platform.

## Purpose

This project explores document processing, information retrieval, embeddings, and Retrieval-Augmented Generation through a practical Python application.

