# RAG-Powered Complaint Analysis Chatbot

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
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning utilities
│   ├── chunking_embedding.py   # Text chunking and embedding
│   ├── vector_store.py         # ChromaDB operations
│   └── rag_pipeline.py         # RAG core logic (Task 3)
├── tests/
│   └── __init__.py
├── app.py                      # Gradio/Streamlit UI (Task 4)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- ~6GB disk space for data

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/FEBEN-G/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download data**
Place the following files in `data/raw/`:
- `complaints.csv` - Full CFPB dataset
- `complaint_embeddings.parquet` - Pre-built embeddings

## 📊 Tasks

### ✅ Task 1: EDA and Data Preprocessing
**Notebook:** `notebooks/task1_eda_preprocessing.ipynb`

- Load and explore CFPB complaint dataset
- Analyze product distribution and narrative characteristics
- Filter for target products
- Clean text narratives
- Save processed data

**Output:** `data/processed/filtered_complaints.csv`

### ✅ Task 2: Text Chunking, Embedding, and Vector Store
**Notebook:** `notebooks/task2_chunking_embedding.ipynb`

- Create stratified sample (~12K complaints)
- Implement text chunking (500 chars, 50 overlap)
- Generate embeddings using `all-MiniLM-L6-v2`
- Build ChromaDB vector store

**Output:** `vector_store/` (persisted collection)

### 🔄 Task 3: RAG Core Logic and Evaluation
**Module:** `src/rag_pipeline.py`

- Load pre-built full-scale vector store
- Implement retriever (top-k similarity search)
- Design prompt template
- Integrate LLM generator
- Qualitative evaluation (5-10 test questions)

### 🔄 Task 4: Interactive Chat Interface
**File:** `app.py`

- Build Gradio/Streamlit UI
- Display AI answers with source citations
- Implement streaming responses
- Add conversation reset

## 🛠️ Usage

### Running Notebooks
```bash
jupyter notebook notebooks/
```

### Running the Chatbot (after Task 4)
```bash
python app.py
```

## 📈 Technical Specifications

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Size:** ~80MB
- **Rationale:** Lightweight, efficient, optimized for semantic similarity

### Chunking Strategy
- **Chunk Size:** 500 characters
- **Overlap:** 50 characters
- **Splitter:** RecursiveCharacterTextSplitter
- **Rationale:** Balances context preservation with embedding quality

### Vector Store
- **Database:** ChromaDB
- **Features:** Persistent storage, metadata filtering, semantic search
- **Metadata:** complaint_id, product, issue, sub_issue, company, state, date

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_preprocessing.py
```

## 📝 Documentation

- [Notebooks README](notebooks/README.md)
- [Implementation Plan](docs/implementation_plan.md)
- [Task List](docs/task.md)

## 🤝 Contributing

This is a training project for 10 Academy. For questions or issues:
1. Check existing documentation
2. Review notebook outputs
3. Contact facilitators

## 📅 Timeline

- **Interim Submission:** Sunday, Jan 4, 2026 (8:00 PM UTC)
  - Tasks 1-2 completed
- **Final Submission:** Tuesday, Jan 13, 2026 (8:00 PM UTC)
  - All tasks completed + final report

## 📄 License

This project is part of the 10 Academy training program.

## 🙏 Acknowledgments

- Consumer Financial Protection Bureau (CFPB) for the dataset
- 10 Academy facilitators: Kerod, Mahbubah, Filimon, Smegnsh
- LangChain and Sentence Transformers communities

---

**Author:** Feben G.  
**GitHub:** [@FEBEN-G](https://github.com/FEBEN-G)  
**Project:** Intelligent Complaint Analysis for Financial Services
