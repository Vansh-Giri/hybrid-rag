# Hybrid RAG System for Technical Knowledge Bases

A RAG pipeline designed to process, search, and synthesize technical documentation.  
This system combines the semantic understanding of dense embeddings (FAISS) with the exact-keyword precision of sparse retrieval (BM25), feeding highly relevant context to a local Large Language Model for factual answer generation.

---

## Key Features


- **Hybrid Retrieval Engine**
  - Dense Retrieval → `all-MiniLM-L6-v2` + FAISS
  - Sparse Retrieval → BM25 keyword index
  - Hybrid Fusion → Min-Max score normalization + weighted fusion

- **Smart Chunking**
  - Recursive chunking
  - Overlap support
  - Context-safe splitting for large PDFs

- **Hardware-Optimized LLM**
  - Ollama local inference
  - Model: `phi4-mini`
  - Context limited to 4096 tokens

- **Modular Architecture**
  - FastAPI backend
  - Streamlit UI
  - Separate ingestion / retrieval / rag modules

- **Persistent Storage**
  - FAISS index saved to disk
  - BM25 index saved to disk
  - No re-indexing after restart

- **Evaluation Pipeline**
  - Precision / Recall
  - Latency measurement

---

## Project Structure

```text
hybrid-rag/
├── api/
│   └── main.py
├── data/
├── evaluation/
│   └── evaluator.py
├── ingestion/
│   ├── chunker.py
│   ├── cleaner.py
│   └── loader.py
├── rag/
│   └── generator.py
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   └── hybrid.py
├── ui/
│   └── app.py
├── vectorstore/
└── config/
```

---

## Tech Stack

- Backend → Python, FastAPI, Uvicorn
- Frontend → Streamlit
- LLM → Ollama (`phi4-mini`)
- Embeddings → sentence-transformers (`all-MiniLM-L6-v2`)
- Vector DB → FAISS (CPU)
- Sparse Retrieval → rank-bm25
- PDF Parsing → PyPDF2 / PyMuPDF
- Text Cleaning → Regex

---

## Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/Vansh-Giri/hybrid-rag.git
cd hybrid-rag
```

### 2. Create virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download Ollama

https://ollama.com

Pull model

```bash
ollama pull phi4-mini
```

Test

```bash
ollama run phi4-mini
```

---

## Running the System

Start backend

```bash
uvicorn api.main:app --reload
```

Start UI

```bash
streamlit run ui/app.py
```

Open browser

```
http://localhost:8501
```

---

## Usage Flow

1. Put PDFs in `data/`
2. Click **Index Documents**
3. Ask questions
4. Hybrid retrieval runs
5. LLM generates answer from context

---

## Evaluation

Evaluation script:

```
evaluation/evaluator.py
```

Metrics measured

- Precision@k
- Recall@k
- Latency
- Memory usage

---

## Target Hardware

- GPU → GTX 1660 Ti (6GB VRAM)
- RAM → 16GB
- CPU → 6 core+

Average latency

- Retrieval → ~0.05s
- Generation → 3–8s

---

## Guardrails

- Strict prompt template
- Context-only answers
- Refuses out-of-scope queries
- Prevents hallucination

---

## Project Info

Developed as Major Project  
Bachelor of Technology  
Computer & Communication Engineering  

Manipal University Jaipur

Author: Vansh Giri  