# Getting Started Guide

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- ~10GB disk space for data and vector store

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/FEBEN-G/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Setup

### Download Required Data

You need two data files:

1. **Full CFPB Dataset** (`complaints.csv`)
   - Place in: `data/raw/complaints.csv`
   
2. **Pre-built Embeddings** (`complaint_embeddings.parquet`)
   - Place in: `data/raw/complaint_embeddings.parquet`

## Running the Project

### Option 1: Run Notebooks (Recommended for Learning)

```bash
# Install Jupyter
pip install jupyter

# Launch Jupyter
jupyter notebook notebooks/
```

**Execute notebooks in order:**
1. `task1_eda_preprocessing.ipynb` - Data exploration and cleaning
2. `task2_chunking_embedding.ipynb` - Create sample vector store
3. `task3_rag_pipeline.ipynb` - Build and test RAG pipeline

### Option 2: Create Vector Store and Run App (Production)

#### Step 1: Create Vector Store from Pre-built Embeddings

```bash
python src/load_prebuilt_embeddings.py
```

This will:
- Load the 2.3GB parquet file
- Create ChromaDB vector store
- Index ~1.37M chunks
- Save to `vector_store/` directory

**Note:** This process takes 10-30 minutes depending on your system.

#### Step 2: Launch the Chatbot

```bash
python app.py
```

The app will be available at: `http://localhost:7860`

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_preprocessing.py

# Run tests in verbose mode
pytest -v
```

## Project Structure

```
rag-complaint-chatbot/
├── data/
│   ├── raw/                    # Raw data files (not in git)
│   │   ├── complaints.csv
│   │   ├── complaint_embeddings.parquet
│   │   └── .gitkeep
│   └── processed/              # Processed data
│       └── filtered_complaints.csv
├── vector_store/               # ChromaDB storage (not in git)
├── notebooks/                  # Jupyter notebooks
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_chunking_embedding.ipynb
│   ├── task3_rag_pipeline.ipynb
│   └── README.md
├── src/                        # Source code
│   ├── data_preprocessing.py
│   ├── chunking_embedding.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── load_prebuilt_embeddings.py
├── tests/                      # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_chunking_embedding.py
│   └── test_rag_pipeline.py
├── app.py                      # Gradio UI
├── requirements.txt
└── README.md
```

## Usage Examples

### Using the RAG Pipeline Programmatically

```python
from src.rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(
    vector_store_path='vector_store',
    collection_name='complaint_embeddings_full'
)

# Ask a question
response = rag.generate_answer(
    query="What are the main issues with credit cards?",
    n_results=5,
    product_filter="credit"
)

# Print answer
print(response['answer'])

# View sources
for source in response['sources']:
    print(f"Product: {source['metadata']['product']}")
    print(f"Issue: {source['metadata']['issue']}")
    print(f"Text: {source['text'][:200]}...")
    print()
```

### Running Evaluation

```python
from src.rag_pipeline import RAGPipeline, EvaluationFramework

rag = RAGPipeline()
evaluator = EvaluationFramework(rag)

# Add test questions
evaluator.add_test_question(
    "Why are customers unhappy with credit cards?",
    product_filter="credit"
)

# Run evaluation
results = evaluator.run_evaluation(n_results=5)
evaluator.print_evaluation_report()
```

## Troubleshooting

### Issue: "Collection not found"

**Solution:** You need to create the vector store first:
```bash
python src/load_prebuilt_embeddings.py
```

### Issue: "Module not found"

**Solution:** Make sure you're in the project directory and virtual environment is activated:
```bash
cd rag-complaint-chatbot
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Out of memory during vector store creation

**Solution:** The process requires ~8GB RAM. Close other applications or use a machine with more RAM.

### Issue: Slow embedding generation

**Solution:** This is normal. The pre-built embeddings are provided to skip this step. Use `load_prebuilt_embeddings.py` instead of generating from scratch.

## Development

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Check code style
flake8 src/ tests/
```

### Running Linters

```bash
flake8 src/ --max-line-length=88
```

### Adding New Tests

1. Create test file in `tests/` directory
2. Name it `test_<module_name>.py`
3. Write test classes and functions
4. Run with `pytest`

## CI/CD

The project includes GitHub Actions workflow for automated testing:
- `.github/workflows/unittests.yml`

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

## Next Steps

1. ✅ Complete data setup
2. ✅ Create vector store
3. ✅ Run the chatbot
4. ✅ Explore notebooks for learning
5. ✅ Run tests to verify installation

## Support

For issues or questions:
- Check the [README.md](README.md)
- Review the notebooks
- Examine test files for usage examples
- Check GitHub Issues

## License

This project is part of the 10 Academy training program.
