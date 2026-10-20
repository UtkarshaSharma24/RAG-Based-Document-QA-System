# RAG-Based Document Q&A System

## 📌 Project Overview

The RAG-Based Document Q&A System is a Python application that allows users to ask questions about a collection of documents and receive relevant answers based on the available content.

The system uses Retrieval-Augmented Generation (RAG) concepts to retrieve relevant document sections before generating or extracting an answer. This helps users find information without manually reading every document.

## 🎯 Objectives

- Build a document-based question-answering system.
- Extract and process text from different file formats.
- Split documents into smaller chunks for efficient retrieval.
- Retrieve relevant information using embeddings or TF-IDF.
- Generate answers based on retrieved content.
- Evaluate retrieval performance using predefined questions.

## ✨ Features

- Document loading from TXT, Markdown, PDF, and DOCX files.
- Text chunking and preprocessing.
- Semantic retrieval using embeddings.
- TF-IDF-based retrieval fallback.
- Extractive answer generation.
- Optional integration with external LLM providers.
- Streamlit-based user interface.
- Evaluation metrics for retrieval performance.
- Exploratory Data Analysis (EDA) and report generation.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Document Processing:** PyPDF, python-docx
- **Machine Learning:** Scikit-learn, NumPy, Pandas
- **Embeddings:** Sentence Transformers / TF-IDF
- **Vector Database:** ChromaDB
- **LLM Integration:** OpenAI / Ollama (optional)
- **Data Visualization:** Matplotlib, Seaborn

## 📂 Project Structure

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

*Note: The exact contents of the `figures/`, `reports/`, and `data/corpus/` folders depend on the files included in the project.*

## ⚙️ Requirements

- Python 3.10 or later
- pip
- Git (optional, for version control)

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Based_Document_QA_System
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

If you have already downloaded the project, open a terminal in the project folder.

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

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

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## 💬 How to Use

1. Start the Streamlit application.
2. Open the application in your browser.
3. Enter a question related to the available documents.
4. Allow the system to retrieve relevant document content.
5. Review the answer and supporting information displayed by the application.
6. Try additional questions to explore the document collection.

The quality of the answer depends on the available documents, retrieval settings, and the relevance of the retrieved content.

## 🔍 Retrieval Methods

### Embedding-Based Retrieval

The system can use text embeddings to represent document chunks and retrieve content that is semantically similar to a user's question.

### TF-IDF Retrieval

TF-IDF provides a lightweight alternative based on the importance of words in the document collection and the user's query.

To run with the TF-IDF configuration in Windows PowerShell:

```powershell
$env:EMBEDDER = "tfidf"
python -m streamlit run app.py
```

The supported configuration depends on the settings defined in `rag/config.py`.

## 📊 Evaluation

The project includes an evaluation script for testing retrieval and answer performance using predefined questions.

Run the evaluation script:

```bash
python evaluate.py
```

To run evaluation with TF-IDF:

```powershell
$env:EMBEDDER = "tfidf"
python evaluate.py
```

For exploratory data analysis, run:

```bash
python eda.py
```

Depending on the configuration and execution, the scripts may generate evaluation results and analysis outputs under the reports or data directories.

### Evaluation Metrics

- **Hit@1:** Checks whether a relevant document is ranked first.
- **Hit@3:** Checks whether a relevant document appears among the top three results.
- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant result is ranked.
- **Answer Accuracy:** Evaluates answers against the project's predefined evaluation criteria.
- **Refusal Rate:** Measures how often the system declines to answer.
- **Latency:** Measures the time required to process a query.

Actual metric values should be reported only after running the evaluation scripts.

## 🔐 Optional LLM Configuration

The application may support external LLM providers depending on the configuration in `rag/llm.py`.

For example, if OpenAI is supported by your configured backend, set the required environment variables in PowerShell:

```powershell
$env:LLM_BACKEND = "openai"
$env:OPENAI_API_KEY = "YOUR_API_KEY_HERE"
python -m streamlit run app.py
```

Replace the placeholder with your own API key. Never publish API keys or other secrets in your GitHub repository.

The application may also use an extractive fallback that returns information from retrieved document content without requiring an external generative model.

## 🧪 Dataset and Documents

The project contains a document corpus and evaluation datasets used for retrieval experiments.

The supplied documents and evaluation questions are intended for project demonstration and testing. Check the dataset files and data dictionary before making claims about the dataset size, origin, or quality.

If synthetic data is included, it should be clearly identified as synthetic and should not be presented as real-world user data.

## ⚠️ Limitations

- Answers depend on the quality and coverage of the source documents.
- Retrieval may return irrelevant content for ambiguous or unsupported questions.
- Embedding models may require an internet connection on the first download.
- Performance depends on the selected retrieval method and system configuration.
- External LLM providers may require API keys and may incur usage costs.
- Evaluation results depend on the quality of the test questions and evaluation criteria.

## 🔮 Future Improvements

- Add support for more document formats.
- Improve chunking and retrieval strategies.
- Add reranking for more relevant search results.
- Improve source citations and answer traceability.
- Expand the evaluation dataset.
- Add user authentication and document management.
- Deploy the application to a cloud platform.

## 👨‍💻 Author

**Utkarsh Sharma**  

Project: RAG-Based Document Q&A System

---

*This project was developed for educational and learning purposes to explore document retrieval, embeddings, and Retrieval-Augmented Generation (RAG).*
