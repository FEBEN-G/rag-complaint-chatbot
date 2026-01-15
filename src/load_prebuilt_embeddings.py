"""
Script to load pre-built embeddings from parquet and create ChromaDB vector store.
This creates the full-scale vector store for Task 3 and Task 4.
"""
import pandas as pd
import numpy as np
import chromadb
from pathlib import Path
from tqdm import tqdm
import ast


def load_embeddings_from_parquet(parquet_path: str) -> pd.DataFrame:
    """Load the pre-built embeddings parquet file."""
    print(f"Loading embeddings from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} chunks")
    print(f"Columns: {df.columns.tolist()}")
    return df


def create_vector_store(
    df: pd.DataFrame,
    vector_store_path: str = 'vector_store',
    collection_name: str = 'complaint_embeddings_full',
    batch_size: int = 1000
):
    """
    Create ChromaDB vector store from embeddings dataframe.
    
    Args:
        df: DataFrame with embeddings and metadata
        vector_store_path: Path to store the vector database
        collection_name: Name of the collection
        batch_size: Batch size for insertion
    """
    # Create vector store directory
    vs_path = Path(vector_store_path)
    vs_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInitializing ChromaDB at: {vs_path}")
    client = chromadb.PersistentClient(path=str(vs_path))
    
    # Delete existing collection if present
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Full complaint embeddings for production RAG"}
    )
    print(f"Created collection: {collection_name}")
    
    # Prepare data
    print("\nPreparing data for indexing...")
    
    # Extract embeddings (handle different possible formats)
    if 'embedding' in df.columns:
        embeddings_col = 'embedding'
    elif 'embeddings' in df.columns:
        embeddings_col = 'embeddings'
    else:
        # Find column with list/array data
        for col in df.columns:
            if isinstance(df[col].iloc[0], (list, np.ndarray)):
                embeddings_col = col
                break
    
    print(f"Using embedding column: {embeddings_col}")
    
    # Convert embeddings to list format if needed
    embeddings = []
    for emb in tqdm(df[embeddings_col], desc="Processing embeddings"):
        if isinstance(emb, str):
            # Parse string representation
            emb = ast.literal_eval(emb)
        elif isinstance(emb, np.ndarray):
            emb = emb.tolist()
        embeddings.append(emb)
    
    # Prepare documents (text chunks)
    if 'text' in df.columns:
        documents = df['text'].tolist()
    elif 'chunk' in df.columns:
        documents = df['chunk'].tolist()
    elif 'chunk_text' in df.columns:
        documents = df['chunk_text'].tolist()
    else:
        print("Warning: No text column found, using empty strings")
        documents = [""] * len(df)
    
    # Prepare metadata
    metadata_cols = [
        'complaint_id', 'product_category', 'product', 'issue', 
        'sub_issue', 'company', 'state', 'date_received',
        'chunk_index', 'total_chunks'
    ]
    
    metadatas = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing metadata"):
        metadata = {}
        for col in metadata_cols:
            if col in df.columns:
                value = row[col]
                # Convert to string for ChromaDB
                if pd.notna(value):
                    metadata[col] = str(value)
                else:
                    metadata[col] = "Unknown"
        metadatas.append(metadata)
    
    # Generate IDs
    ids = [f"chunk_{i}" for i in range(len(df))]
    
    # Add to collection in batches
    print(f"\nIndexing {len(ids):,} chunks in batches of {batch_size}...")
    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing batches"):
        batch_end = min(i + batch_size, len(ids))
        
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    print(f"\n✅ Successfully created vector store!")
    print(f"   Collection: {collection_name}")
    print(f"   Total chunks: {collection.count():,}")
    print(f"   Location: {vs_path}")
    
    return collection


def main():
    """Main function to create the vector store."""
    # Paths
    parquet_path = 'data/raw/complaint_embeddings.parquet'
    vector_store_path = 'vector_store'
    
    # Load embeddings
    df = load_embeddings_from_parquet(parquet_path)
    
    # Show sample
    print("\nSample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    
    # Create vector store
    collection = create_vector_store(
        df=df,
        vector_store_path=vector_store_path,
        collection_name='complaint_embeddings_full',
        batch_size=1000
    )
    
    print("\n" + "="*80)
    print("Vector store creation complete!")
    print("="*80)
    print("\nYou can now use the RAG pipeline with this vector store.")
    print("Example:")
    print("  from src.rag_pipeline import RAGPipeline")
    print("  rag = RAGPipeline()")
    print("  response = rag.generate_answer('Why are customers unhappy with credit cards?')")


if __name__ == "__main__":
    main()
