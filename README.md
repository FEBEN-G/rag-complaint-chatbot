# RAG-Powered Complaint Analysis Chatbot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/FEBEN-G/rag-complaint-chatbot/workflows/Unit%20Tests/badge.svg)](https://github.com/FEBEN-G/rag-complaint-chatbot/actions)

An intelligent complaint analysis system for CrediTrust Financial that transforms customer feedback into actionable insights using Retrieval-Augmented Generation (RAG).

## 🎯 Project Overview

This project builds an AI-powered chatbot that enables internal stakeholders (Product Managers, Support Teams, Compliance) to query customer complaints using natural language and receive synthesized, evidence-backed answers in seconds.

### Key Features
- **Semantic Search**: Find relevant complaints using vector similarity
- **Multi-Product Support**: Analyze complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers
- **RAG Pipeline**: Combines retrieval with LLM generation for accurate, contextual answers
- **Interactive UI**: User-friendly Gradio/Streamlit interface with source citation

## 📁 Project Structure

```
rag-complaint-chatbot/
├── data/
│   ├── raw/                    # Raw CFPB complaint data
│   └── processed/              # Cleaned and filtered data
├── vector_store/               # Persisted ChromaDB index
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_chunking_embedding.ipynb
│   └── task3_rag_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning utilities
│   ├── chunking_embedding.py   # Text chunking and embedding
│   ├── vector_store.py         # ChromaDB operations
│   ├── rag_pipeline.py         # RAG core logic
│   ├── load_prebuilt_embeddings.py
│   └── inspect_embeddings.py
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_chunking_embedding.py
│   └── test_rag_pipeline.py
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .vscode/
│   └── settings.json
├── app.py                      # Gradio UI application
├── requirements.txt
├── setup.cfg
└── README.md
```

## 🚀 Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/FEBEN-G/rag-complaint-chatbot.git
   cd rag-complaint-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Data**
   Place `complaints.csv` and `complaint_embeddings.parquet` in `data/raw/`.

5. **Initialize Vector Store**
   ```bash
   python src/load_prebuilt_embeddings.py
   ```

### Usage

**Run the Chatbot:**
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

**Run Tests:**
```bash
pytest
```

## 📞 Contact

**Author:** Feben G.  
**GitHub:** [@FEBEN-G](https://github.com/FEBEN-G)  
**Project:** Intelligent Complaint Analysis for Financial Services
