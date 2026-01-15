"""
Text chunking and embedding utilities.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ChunkingService:
    """Service for chunking complaint narratives."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunking service.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_chunks_with_metadata(self, df: pd.DataFrame, text_col: str = 'cleaned_narrative') -> List[Dict]:
        """
        Create text chunks with associated metadata.
        
        Args:
            df: DataFrame containing complaints
            text_col: Name of the text column to chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        all_chunks = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating chunks"):
            narrative = row[text_col]
            
            # Skip if narrative is empty
            if not narrative or len(str(narrative).strip()) == 0:
                continue
            
            # Split into chunks
            chunks = self.text_splitter.split_text(str(narrative))
            
            # Create metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_data = {
                    'text': chunk,
                    'complaint_id': str(row.get('complaint_id', idx)),
                    'product': row.get('Product', 'Unknown'),
                    'issue': row.get('Issue', 'Unknown'),
                    'sub_issue': row.get('Sub-issue', 'Unknown'),
                    'company': row.get('Company', 'Unknown'),
                    'state': row.get('State', 'Unknown'),
                    'date_received': str(row.get('Date received', 'Unknown')),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_data)
        
        return all_chunks


class EmbeddingService:
    """Service for generating embeddings."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array embedding
        """
        return self.model.encode(text, convert_to_numpy=True)
