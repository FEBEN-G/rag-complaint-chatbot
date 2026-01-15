"""
Data preprocessing utilities for complaint analysis.
"""
import re
import pandas as pd
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean complaint narrative text.
    
    Args:
        text: Raw complaint narrative
        
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r'i am writing to file a complaint',
        r'i would like to file a complaint',
        r'this is a complaint about',
        r'xxxx',  # Common redaction marker
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep only alphanumeric characters and basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?;:\-\'\"()]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def filter_target_products(df: pd.DataFrame, product_col: str = 'Product') -> pd.DataFrame:
    """
    Filter dataset for target financial products.
    
    Args:
        df: Input dataframe
        product_col: Name of the product column
        
    Returns:
        Filtered dataframe
    """
    def matches_target_product(product):
        if pd.isna(product):
            return False
        product_lower = str(product).lower()
        return any([
            'credit card' in product_lower,
            'personal loan' in product_lower,
            'savings account' in product_lower,
            'money transfer' in product_lower
        ])
    
    return df[df[product_col].apply(matches_target_product)].copy()


def prepare_complaints_data(
    df: pd.DataFrame,
    narrative_col: str = 'Consumer complaint narrative',
    clean_narratives: bool = True
) -> pd.DataFrame:
    """
    Prepare complaints data for RAG pipeline.
    
    Args:
        df: Input dataframe
        narrative_col: Name of the narrative column
        clean_narratives: Whether to clean the narratives
        
    Returns:
        Prepared dataframe
    """
    # Filter for target products
    df_filtered = filter_target_products(df)
    
    # Remove empty narratives
    df_filtered = df_filtered[df_filtered[narrative_col].notna()].copy()
    
    # Clean narratives if requested
    if clean_narratives:
        df_filtered['cleaned_narrative'] = df_filtered[narrative_col].apply(clean_text)
        # Remove rows with empty cleaned narratives
        df_filtered = df_filtered[df_filtered['cleaned_narrative'].str.len() > 0].copy()
    
    return df_filtered
