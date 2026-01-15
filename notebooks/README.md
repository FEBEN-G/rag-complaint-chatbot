# Notebooks

This directory contains Jupyter notebooks for the RAG Complaint Chatbot project.

## Notebooks Overview

### Task 1: EDA and Data Preprocessing
**File:** `task1_eda_preprocessing.ipynb`

This notebook performs exploratory data analysis and preprocessing on the CFPB complaint dataset:
- Loads and explores the raw complaint data
- Analyzes product distribution and narrative characteristics
- Filters for target products (Credit Card, Personal Loan, Savings Account, Money Transfers)
- Cleans text narratives (lowercasing, removing special characters, boilerplate removal)
- Saves the filtered and cleaned dataset

**Output:** `../data/processed/filtered_complaints.csv`

### Task 2: Text Chunking, Embedding, and Vector Store Indexing
**File:** `task2_chunking_embedding.ipynb`

This notebook creates a searchable vector index from complaint narratives:
- Creates a stratified sample of ~12,000 complaints
- Implements text chunking using RecursiveCharacterTextSplitter
- Generates embeddings using `all-MiniLM-L6-v2`
- Builds a ChromaDB vector store with metadata
- Tests semantic search functionality

**Output:** `../vector_store/` (persisted ChromaDB collection)

## Running the Notebooks

### Prerequisites
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ../requirements.txt

# Install Jupyter
pip install jupyter
```

### Launch Jupyter
```bash
# From the project root
jupyter notebook notebooks/
```

### Execution Order
1. Run `task1_eda_preprocessing.ipynb` first
2. Then run `task2_chunking_embedding.ipynb`

## Data Requirements

Place the following files in `../data/raw/`:
- `complaints.csv` - Full CFPB complaint dataset
- `complaint_embeddings.parquet` - Pre-built embeddings (for Task 3+)

## Notes

- Task 1 processes the full dataset and creates a filtered version
- Task 2 works with a stratified sample for training purposes
- For production (Task 3-4), use the pre-built embeddings from `complaint_embeddings.parquet`
