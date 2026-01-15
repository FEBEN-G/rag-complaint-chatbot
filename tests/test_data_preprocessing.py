"""
Unit tests for data preprocessing module.
"""
import pytest
import pandas as pd
from src.data_preprocessing import clean_text, filter_target_products, prepare_complaints_data


class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "I AM WRITING TO FILE A COMPLAINT about my credit card."
        result = clean_text(text)
        assert result == "about my credit card."
    
    def test_clean_text_removes_urls(self):
        """Test URL removal."""
        text = "Visit http://example.com for more info"
        result = clean_text(text)
        assert "http" not in result
        assert "example.com" not in result
    
    def test_clean_text_removes_emails(self):
        """Test email removal."""
        text = "Contact me at test@example.com"
        result = clean_text(text)
        assert "@" not in result
        assert "test@example.com" not in result
    
    def test_clean_text_removes_special_chars(self):
        """Test special character removal."""
        text = "This has #special @characters!"
        result = clean_text(text)
        assert "#" not in result
        assert "@" not in result
    
    def test_clean_text_handles_none(self):
        """Test handling of None input."""
        result = clean_text(None)
        assert result == ""
    
    def test_clean_text_removes_extra_whitespace(self):
        """Test whitespace normalization."""
        text = "Too    many     spaces"
        result = clean_text(text)
        assert "  " not in result


class TestFilterTargetProducts:
    """Tests for the filter_target_products function."""
    
    def test_filter_credit_card(self):
        """Test filtering for credit card products."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Mortgage', 'Credit card or prepaid card']
        })
        result = filter_target_products(df)
        assert len(result) == 2
        assert all('credit card' in p.lower() for p in result['Product'])
    
    def test_filter_personal_loan(self):
        """Test filtering for personal loan products."""
        df = pd.DataFrame({
            'Product': ['Personal loan', 'Mortgage', 'Student loan']
        })
        result = filter_target_products(df)
        assert len(result) == 1
        assert 'personal loan' in result['Product'].iloc[0].lower()
    
    def test_filter_savings_account(self):
        """Test filtering for savings account products."""
        df = pd.DataFrame({
            'Product': ['Savings account', 'Checking account', 'Money market account']
        })
        result = filter_target_products(df)
        assert len(result) == 1
    
    def test_filter_money_transfer(self):
        """Test filtering for money transfer products."""
        df = pd.DataFrame({
            'Product': ['Money transfer', 'Money transfers', 'Wire transfer']
        })
        result = filter_target_products(df)
        assert len(result) == 2
    
    def test_filter_excludes_others(self):
        """Test that non-target products are excluded."""
        df = pd.DataFrame({
            'Product': ['Mortgage', 'Student loan', 'Debt collection']
        })
        result = filter_target_products(df)
        assert len(result) == 0


class TestPrepareComplaintsData:
    """Tests for the prepare_complaints_data function."""
    
    def test_prepare_removes_empty_narratives(self):
        """Test that empty narratives are removed."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Credit card'],
            'Consumer complaint narrative': ['Valid complaint', None]
        })
        result = prepare_complaints_data(df)
        assert len(result) == 1
    
    def test_prepare_cleans_narratives(self):
        """Test that narratives are cleaned."""
        df = pd.DataFrame({
            'Product': ['Credit card'],
            'Consumer complaint narrative': ['I AM WRITING TO FILE A COMPLAINT']
        })
        result = prepare_complaints_data(df, clean_narratives=True)
        assert 'cleaned_narrative' in result.columns
        assert len(result['cleaned_narrative'].iloc[0]) > 0
    
    def test_prepare_filters_products(self):
        """Test that products are filtered."""
        df = pd.DataFrame({
            'Product': ['Credit card', 'Mortgage', 'Personal loan'],
            'Consumer complaint narrative': ['complaint 1', 'complaint 2', 'complaint 3']
        })
        result = prepare_complaints_data(df)
        assert len(result) == 2
        assert 'Mortgage' not in result['Product'].values
