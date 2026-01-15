"""
Unit tests for RAG pipeline module.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.rag_pipeline import RAGPipeline, EvaluationFramework


class TestRAGPipeline:
    """Tests for the RAGPipeline class."""
    
    @patch('src.rag_pipeline.SentenceTransformer')
    @patch('src.rag_pipeline.chromadb.PersistentClient')
    def test_rag_pipeline_initialization(self, mock_chroma, mock_transformer):
        """Test RAG pipeline initialization."""
        # Mock the collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1000
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        rag = RAGPipeline(
            vector_store_path='test_store',
            collection_name='test_collection'
        )
        
        assert rag.collection is not None
        assert rag.embedding_model is not None
    
    def test_format_context(self):
        """Test context formatting."""
        rag = RAGPipeline.__new__(RAGPipeline)  # Create without __init__
        
        results = {
            'documents': [[
                'First complaint text',
                'Second complaint text'
            ]],
            'metadatas': [[
                {'product': 'Credit card', 'issue': 'Billing'},
                {'product': 'Personal loan', 'issue': 'Payment'}
            ]]
        }
        
        context = rag.format_context(results)
        assert 'Credit card' in context
        assert 'Personal loan' in context
        assert 'First complaint text' in context
    
    def test_generate_prompt(self):
        """Test prompt generation."""
        rag = RAGPipeline.__new__(RAGPipeline)
        
        query = "What are the main issues?"
        context = "Sample context"
        
        prompt = rag.generate_prompt(query, context)
        assert query in prompt
        assert context in prompt
        assert 'financial analyst' in prompt.lower()
    
    def test_synthesize_from_retrieval(self):
        """Test answer synthesis from retrieval."""
        rag = RAGPipeline.__new__(RAGPipeline)
        
        results = {
            'documents': [[
                'Complaint about billing',
                'Issue with payment'
            ]],
            'metadatas': [[
                {'product': 'Credit card', 'issue': 'Billing'},
                {'product': 'Credit card', 'issue': 'Payment'}
            ]],
            'distances': [[0.3, 0.4]]
        }
        
        answer = rag._synthesize_from_retrieval(results)
        assert len(answer) > 0
        assert 'Credit card' in answer or 'credit card' in answer.lower()


class TestEvaluationFramework:
    """Tests for the EvaluationFramework class."""
    
    def test_add_test_question(self):
        """Test adding test questions."""
        mock_rag = MagicMock()
        evaluator = EvaluationFramework(mock_rag)
        
        evaluator.add_test_question(
            question="Test question?",
            expected_themes=['billing', 'payment'],
            product_filter='credit'
        )
        
        assert len(evaluator.test_questions) == 1
        assert evaluator.test_questions[0]['question'] == "Test question?"
        assert 'billing' in evaluator.test_questions[0]['expected_themes']
    
    @patch.object(RAGPipeline, 'generate_answer')
    def test_run_evaluation(self, mock_generate):
        """Test running evaluation."""
        mock_rag = MagicMock()
        mock_generate.return_value = {
            'answer': 'Test answer',
            'n_sources': 5,
            'sources': [
                {
                    'metadata': {'product': 'Credit card', 'issue': 'Billing'},
                    'distance': 0.3
                }
            ]
        }
        mock_rag.generate_answer = mock_generate
        
        evaluator = EvaluationFramework(mock_rag)
        evaluator.add_test_question("Test question?")
        
        results_df = evaluator.run_evaluation(n_results=5)
        
        assert len(results_df) == 1
        assert 'question' in results_df.columns
        assert 'answer' in results_df.columns
