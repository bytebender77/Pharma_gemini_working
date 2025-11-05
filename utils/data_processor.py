"""
Data processing and formatting utilities.
"""

import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger("pharma_ai.data_processor")

class DataProcessor:
    """Process and format data from various sources."""
    
    @staticmethod
    def format_clinical_trials(trials: List[Dict]) -> pd.DataFrame:
        """
        Format clinical trials data into DataFrame.
        
        Args:
            trials: List of trial dictionaries
        
        Returns:
            Formatted DataFrame
        """
        if not trials:
            return pd.DataFrame()
        
        df = pd.DataFrame(trials)
        
        # Select relevant columns
        columns = ['nct_id', 'title', 'status', 'phase', 'conditions']
        available_columns = [col for col in columns if col in df.columns]
        
        return df[available_columns]
    
    @staticmethod
    def format_publications(articles: List[Dict]) -> pd.DataFrame:
        """
        Format publication data into DataFrame.
        
        Args:
            articles: List of article dictionaries
        
        Returns:
            Formatted DataFrame
        """
        if not articles:
            return pd.DataFrame()
        
        df = pd.DataFrame(articles)
        
        # Format authors
        if 'authors' in df.columns:
            df['authors'] = df['authors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        return df
    
    @staticmethod
    def summarize_data(data: Any, max_length: int = 500) -> str:
        """
        Summarize data for display.
        
        Args:
            data: Data to summarize
            max_length: Maximum length of summary
        
        Returns:
            Summary string
        """
        if isinstance(data, list):
            summary = f"Found {len(data)} items"
            if data:
                summary += f"\nFirst item: {str(data[0])[:100]}..."
        elif isinstance(data, dict):
            summary = f"Dictionary with {len(data)} keys: {', '.join(list(data.keys())[:5])}"
        else:
            summary = str(data)[:max_length]
        
        return summary
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        Extract top keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords
        
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be enhanced with NLP)
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(words)
        
        return [word for word, count in word_freq.most_common(top_n)]
    
    @staticmethod
    def merge_results(results_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge results from multiple sources.
        
        Args:
            results_dict: Dictionary of results from different sources
        
        Returns:
            Merged results
        """
        merged = {
            'sources': list(results_dict.keys()),
            'total_results': sum(len(v) if isinstance(v, list) else 1 for v in results_dict.values()),
            'data': results_dict
        }
        
        return merged