"""
Vector store utilities for ChromaDB.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from tqdm import tqdm


class VectorStoreService:
    """Service for managing ChromaDB vector store."""
    
    def __init__(self, persist_directory: str = '../vector_store'):
        """
        Initialize vector store service.
        
        Args:
            persist_directory: Directory to persist the vector store
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        self.collection = None
    
    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        reset: bool = False
    ):
        """
        Create or get a collection.
        
        Args:
            name: Collection name
            metadata: Collection metadata
            reset: Whether to delete existing collection
        """
        if reset:
            try:
                self.client.delete_collection(name=name)
                print(f"Deleted existing collection: {name}")
            except:
                pass
        
        self.collection = self.client.create_collection(
            name=name,
            metadata=metadata or {"description": "Complaint embeddings"}
        )
        print(f"Created collection: {name}")
    
    def get_collection(self, name: str):
        """
        Get an existing collection.
        
        Args:
            name: Collection name
        """
        self.collection = self.client.get_collection(name=name)
        print(f"Loaded collection: {name}")
        return self.collection
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, str]],
        ids: Optional[List[str]] = None,
        batch_size: int = 1000
    ):
        """
        Add embeddings to the collection.
        
        Args:
            embeddings: Numpy array of embeddings
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique IDs (generated if not provided)
            batch_size: Batch size for adding
        """
        if self.collection is None:
            raise ValueError("No collection initialized. Call create_collection or get_collection first.")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(documents))]
        
        # Add in batches
        for i in tqdm(range(0, len(ids), batch_size), desc="Indexing batches"):
            batch_end = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        print(f"Successfully indexed {len(ids):,} chunks")
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Query the vector store.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            Query results
        """
        if self.collection is None:
            raise ValueError("No collection initialized.")
        
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        if self.collection is None:
            raise ValueError("No collection initialized.")
        
        return {
            'name': self.collection.name,
            'count': self.collection.count(),
            'metadata': self.collection.metadata
        }
