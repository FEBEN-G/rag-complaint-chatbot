"""
RAG Pipeline for Complaint Analysis.
Implements retrieval and generation logic.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path


class RAGPipeline:
    """
    Complete RAG pipeline for complaint analysis.
    Combines retrieval from vector store with LLM generation.
    """
    
    def __init__(
        self,
        vector_store_path: str = 'vector_store',
        collection_name: str = 'complaint_embeddings_full',
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store_path: Path to ChromaDB vector store
            collection_name: Name of the collection
            embedding_model: Sentence transformer model name
        """
        self.vector_store_path = Path(vector_store_path)
        self.collection_name = collection_name
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        print(f"Connecting to vector store: {vector_store_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_path)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
            print(f"Collection size: {self.collection.count():,} chunks")
        except:
            print(f"Collection '{collection_name}' not found. You need to create it first.")
            self.collection = None
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        product_filter: Optional[str] = None
    ) -> Dict:
        """
        Retrieve relevant complaint chunks for a query.
        
        Args:
            query: User question
            n_results: Number of results to retrieve
            product_filter: Optional product category filter
            
        Returns:
            Dictionary with retrieved documents, metadata, and distances
        """
        if self.collection is None:
            raise ValueError("No collection loaded. Cannot retrieve.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Build metadata filter if product specified
        where_filter = None
        if product_filter:
            where_filter = {"product": {"$contains": product_filter.lower()}}
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def format_context(self, results: Dict) -> str:
        """
        Format retrieved results into context string for LLM.
        
        Args:
            results: Results from retrieve()
            
        Returns:
            Formatted context string
        """
        if not results['documents'] or not results['documents'][0]:
            return "No relevant information found."
        
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        ), 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Product: {metadata.get('product', 'Unknown')}\n"
                f"Issue: {metadata.get('issue', 'Unknown')}\n"
                f"Complaint: {doc}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        """
        Generate the complete prompt for the LLM.
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Complete prompt string
        """
        prompt = f"""You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context.

Instructions:
- Use ONLY the information from the retrieved complaint excerpts below
- Provide a clear, concise answer that synthesizes the main themes
- If the context doesn't contain enough information, state that clearly
- Cite specific issues or patterns you observe
- Be objective and professional

Context (Retrieved Complaint Excerpts):
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(
        self,
        query: str,
        n_results: int = 5,
        product_filter: Optional[str] = None,
        use_llm: bool = False,
        llm_model: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Complete RAG pipeline: retrieve and generate answer.
        
        Args:
            query: User question
            n_results: Number of chunks to retrieve
            product_filter: Optional product filter
            use_llm: Whether to use LLM for generation (requires API/local model)
            llm_model: LLM model to use (e.g., 'gpt-3.5-turbo', 'mistral')
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant chunks
        results = self.retrieve(query, n_results, product_filter)
        
        # Step 2: Format context
        context = self.format_context(results)
        
        # Step 3: Generate prompt
        prompt = self.generate_prompt(query, context)
        
        # Step 4: Generate answer
        if use_llm and llm_model:
            # Use actual LLM (requires additional setup)
            answer = self._call_llm(prompt, llm_model)
        else:
            # Fallback: Return synthesized summary from retrieval
            answer = self._synthesize_from_retrieval(results)
        
        # Step 5: Package response
        response = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'text': doc,
                    'metadata': metadata,
                    'distance': distance
                }
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ],
            'prompt': prompt,
            'n_sources': len(results['documents'][0])
        }
        
        return response
    
    def _synthesize_from_retrieval(self, results: Dict) -> str:
        """
        Create a basic answer from retrieved results (without LLM).
        
        Args:
            results: Retrieval results
            
        Returns:
            Synthesized answer
        """
        if not results['documents'] or not results['documents'][0]:
            return "I couldn't find relevant complaints to answer your question."
        
        # Extract key information
        products = set()
        issues = set()
        
        for metadata in results['metadatas'][0]:
            products.add(metadata.get('product', 'Unknown'))
            issues.add(metadata.get('issue', 'Unknown'))
        
        # Build summary
        summary_parts = [
            f"Based on {len(results['documents'][0])} relevant complaints:",
            f"\nProducts mentioned: {', '.join(products)}",
            f"\nCommon issues: {', '.join(list(issues)[:3])}",
            f"\nKey themes from the complaints:",
        ]
        
        # Add snippets from top 3 results
        for i, doc in enumerate(results['documents'][0][:3], 1):
            snippet = doc[:150] + "..." if len(doc) > 150 else doc
            summary_parts.append(f"\n{i}. {snippet}")
        
        return "\n".join(summary_parts)
    
    def _call_llm(self, prompt: str, model: str) -> str:
        """
        Call an LLM with the prompt.
        
        Args:
            prompt: Complete prompt
            model: Model name
            
        Returns:
            Generated answer
        """
        # This is a placeholder - actual implementation depends on the LLM being used
        # Options:
        # 1. OpenAI API: openai.ChatCompletion.create()
        # 2. Hugging Face: pipeline or transformers
        # 3. Local models: llama.cpp, ollama, etc.
        
        try:
            if 'gpt' in model.lower():
                # OpenAI implementation
                import openai
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            
            elif 'mistral' in model.lower() or 'llama' in model.lower():
                # Hugging Face implementation
                from transformers import pipeline
                generator = pipeline('text-generation', model=model)
                response = generator(prompt, max_length=500, num_return_sequences=1)
                return response[0]['generated_text']
            
            else:
                return "LLM integration not configured for this model."
        
        except Exception as e:
            return f"Error calling LLM: {str(e)}"


class EvaluationFramework:
    """Framework for evaluating RAG pipeline performance."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize evaluation framework.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag = rag_pipeline
        self.test_questions = []
        self.results = []
    
    def add_test_question(
        self,
        question: str,
        expected_themes: Optional[List[str]] = None,
        product_filter: Optional[str] = None
    ):
        """
        Add a test question to the evaluation set.
        
        Args:
            question: Test question
            expected_themes: Expected themes/topics in answer
            product_filter: Product filter for this question
        """
        self.test_questions.append({
            'question': question,
            'expected_themes': expected_themes or [],
            'product_filter': product_filter
        })
    
    def run_evaluation(self, n_results: int = 5) -> pd.DataFrame:
        """
        Run evaluation on all test questions.
        
        Args:
            n_results: Number of results to retrieve per question
            
        Returns:
            DataFrame with evaluation results
        """
        self.results = []
        
        for test in self.test_questions:
            print(f"\nEvaluating: {test['question'][:50]}...")
            
            # Get answer
            response = self.rag.generate_answer(
                query=test['question'],
                n_results=n_results,
                product_filter=test['product_filter']
            )
            
            # Store result
            result = {
                'question': test['question'],
                'answer': response['answer'],
                'n_sources': response['n_sources'],
                'top_product': response['sources'][0]['metadata'].get('product', 'Unknown') if response['sources'] else 'None',
                'top_issue': response['sources'][0]['metadata'].get('issue', 'Unknown') if response['sources'] else 'None',
                'avg_distance': np.mean([s['distance'] for s in response['sources']]) if response['sources'] else 1.0,
                'sources_preview': response['sources'][:2] if response['sources'] else []
            }
            
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def print_evaluation_report(self):
        """Print a formatted evaluation report."""
        if not self.results:
            print("No evaluation results. Run run_evaluation() first.")
            return
        
        print("\n" + "="*80)
        print("RAG PIPELINE EVALUATION REPORT")
        print("="*80)
        
        for i, result in enumerate(self.results, 1):
            print(f"\n{'─'*80}")
            print(f"Question {i}: {result['question']}")
            print(f"{'─'*80}")
            print(f"\nGenerated Answer:")
            print(result['answer'])
            print(f"\nRetrieval Metrics:")
            print(f"  • Sources Retrieved: {result['n_sources']}")
            print(f"  • Average Distance: {result['avg_distance']:.4f}")
            print(f"  • Top Product: {result['top_product']}")
            print(f"  • Top Issue: {result['top_issue']}")
            
            if result['sources_preview']:
                print(f"\nTop Source Previews:")
                for j, source in enumerate(result['sources_preview'], 1):
                    print(f"\n  Source {j} (distance: {source['distance']:.4f}):")
                    print(f"    Product: {source['metadata'].get('product', 'Unknown')}")
                    print(f"    Text: {source['text'][:150]}...")
