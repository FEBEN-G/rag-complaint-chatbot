"""
Unit tests for chunking and embedding module.
"""
import pytest
import pandas as pd
import numpy as np
from src.chunking_embedding import ChunkingService, EmbeddingService


class TestChunkingService:
    """Tests for the ChunkingService class."""
    
    def test_chunking_service_initialization(self):
        """Test ChunkingService initialization."""
        service = ChunkingService(chunk_size=500, chunk_overlap=50)
        assert service.chunk_size == 500
        assert service.chunk_overlap == 50
        assert service.text_splitter is not None
    
    def test_create_chunks_basic(self):
        """Test basic chunk creation."""
        service = ChunkingService(chunk_size=100, chunk_overlap=10)
        df = pd.DataFrame({
            'complaint_id': [1],
            'Product': ['Credit card'],
            'cleaned_narrative': ['This is a test complaint narrative that should be chunked.']
        })
        
        chunks = service.create_chunks_with_metadata(df)
        assert len(chunks) > 0
        assert 'text' in chunks[0]
        assert 'complaint_id' in chunks[0]
        assert 'product' in chunks[0]
    
    def test_create_chunks_with_long_text(self):
        """Test chunking of long text."""
        service = ChunkingService(chunk_size=50, chunk_overlap=10)
        long_text = "word " * 100  # Create long text
        df = pd.DataFrame({
            'complaint_id': [1],
            'Product': ['Credit card'],
            'cleaned_narrative': [long_text]
        })
        
        chunks = service.create_chunks_with_metadata(df)
        assert len(chunks) > 1  # Should create multiple chunks
        assert all('chunk_index' in c for c in chunks)
        assert all('total_chunks' in c for c in chunks)
    
    def test_create_chunks_metadata(self):
        """Test that metadata is properly attached to chunks."""
        service = ChunkingService(chunk_size=100, chunk_overlap=10)
        df = pd.DataFrame({
            'complaint_id': [1],
            'Product': ['Credit card'],
            'Issue': ['Billing dispute'],
            'Company': ['Test Bank'],
            'State': ['CA'],
            'cleaned_narrative': ['Test complaint']
        })
        
        chunks = service.create_chunks_with_metadata(df)
        assert chunks[0]['product'] == 'Credit card'
        assert chunks[0]['issue'] == 'Billing dispute'
        assert chunks[0]['company'] == 'Test Bank'
        assert chunks[0]['state'] == 'CA'


class TestEmbeddingService:
    """Tests for the EmbeddingService class."""
    
    def test_embedding_service_initialization(self):
        """Test EmbeddingService initialization."""
        service = EmbeddingService()
        assert service.model is not None
        assert service.embedding_dim == 384  # all-MiniLM-L6-v2 dimension
    
    def test_encode_single_text(self):
        """Test encoding a single text."""
        service = EmbeddingService()
        text = "This is a test complaint"
        embedding = service.encode_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        service = EmbeddingService()
        texts = [
            "First complaint",
            "Second complaint",
            "Third complaint"
        ]
        embeddings = service.encode(texts, batch_size=2, show_progress=False)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
    
    def test_embedding_similarity(self):
        """Test that similar texts have similar embeddings."""
        service = EmbeddingService()
        text1 = "credit card billing problem"
        text2 = "credit card billing issue"
        text3 = "personal loan application"
        
        emb1 = service.encode_single(text1)
        emb2 = service.encode_single(text2)
        emb3 = service.encode_single(text3)
        
        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
