"""
Tools for accessing PubMed E-utilities API.
"""

import requests
from typing import List, Dict
from crewai.tools import tool
import logging
import time
from config import PUBMED_API, NCBI_API_KEY, RATE_LIMITS
from utils.api_helpers import rate_limit, retry_on_error
from utils.cache_manager import cache

logger = logging.getLogger("pharma_ai.pubmed")

class PubMedAPI:
    """Wrapper for PubMed E-utilities API."""
    
    BASE_URL = PUBMED_API
    
    @staticmethod
    @rate_limit(RATE_LIMITS["pubmed"])
    @retry_on_error(max_attempts=3)
    def search_articles(
        query: str,
        max_results: int = 10,
        sort: str = "relevance"
    ) -> List[str]:
        """
        Search PubMed articles and return PMIDs.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ('relevance', 'date')
        
        Returns:
            List of PubMed IDs (PMIDs)
        """
        # Check cache
        cache_key = f"{query}_{max_results}_{sort}"
        cached_data = cache.get("pubmed_search", cache_key)
        if cached_data:
            return cached_data
        
        search_url = f"{PubMedAPI.BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort
        }
        
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        
        try:
            logger.info(f"Searching PubMed for: {query}")
            
            time.sleep(1 / RATE_LIMITS["pubmed"])
            response = requests.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            logger.info(f"Found {len(pmids)} PubMed articles")
            
            # Cache results
            cache.set("pubmed_search", cache_key, pmids)
            
            return pmids
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    @staticmethod
    @rate_limit(RATE_LIMITS["pubmed"])
    @retry_on_error(max_attempts=3)
    def fetch_article_details(pmids: List[str]) -> List[Dict]:
        """
        Fetch article details for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of article details
        """
        if not pmids:
            return []
        
        # Check cache
        cache_key = "_".join(pmids)
        cached_data = cache.get("pubmed_details", cache_key)
        if cached_data:
            return cached_data
        
        fetch_url = f"{PubMedAPI.BASE_URL}/esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }
        
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        
        try:
            logger.info(f"Fetching details for {len(pmids)} articles")
            
            time.sleep(1 / RATE_LIMITS["pubmed"])
            response = requests.get(fetch_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for pmid in pmids:
                article_data = data.get("result", {}).get(pmid, {})
                
                if article_data and isinstance(article_data, dict):
                    articles.append({
                        "pmid": pmid,
                        "title": article_data.get("title", "N/A"),
                        "authors": [
                            author.get("name", "")
                            for author in article_data.get("authors", [])[:5]
                        ],
                        "journal": article_data.get("fulljournalname", "N/A"),
                        "pub_date": article_data.get("pubdate", "N/A"),
                        "source": article_data.get("source", "N/A"),
                        "volume": article_data.get("volume", "N/A"),
                        "issue": article_data.get("issue", "N/A"),
                        "pages": article_data.get("pages", "N/A"),
                        "doi": article_data.get("elocationid", "N/A"),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
            
            logger.info(f"Retrieved details for {len(articles)} articles")
            
            # Cache results
            cache.set("pubmed_details", cache_key, articles)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching article details: {e}")
            return []
    
    @staticmethod
    def get_article_abstract(pmid: str) -> str:
        """
        Fetch article abstract.
        
        Args:
            pmid: PubMed ID
        
        Returns:
            Article abstract
        """
        fetch_url = f"{PubMedAPI.BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        
        try:
            time.sleep(1 / RATE_LIMITS["pubmed"])
            response = requests.get(fetch_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Simple XML parsing (can be enhanced)
            text = response.text
            
            # Extract abstract text
            import re
            abstract_match = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', text, re.DOTALL)
            
            if abstract_match:
                return abstract_match.group(1)[:1000]
            
            return "Abstract not available"
            
        except Exception as e:
            logger.error(f"Error fetching abstract for PMID {pmid}: {e}")
            return "Abstract not available"


@tool
def search_pubmed_literature(query: str) -> str:
    """
    Search PubMed for scientific articles and publications.
    Use this when you need to find research papers, clinical studies, or scientific evidence.
    
    Args:
        query: Search query (e.g., 'metformin diabetes', 'aspirin cardiovascular disease', 
               'pembrolizumab melanoma')
    
    Returns:
        Formatted string with article information including title, authors, journal, 
        publication date, and PubMed URL
    """
    pmids = PubMedAPI.search_articles(query, max_results=10)
    
    if not pmids:
        return f"No articles found for query: {query}"
    
    articles = PubMedAPI.fetch_article_details(pmids)
    
    if not articles:
        return f"Found {len(pmids)} articles but couldn't retrieve details"
    
    result = f"Found {len(articles)} scientific articles for '{query}':\n\n"
    
    for i, article in enumerate(articles, 1):
        result += f"{i}. {article['title']}\n"
        result += f"   Authors: {', '.join(article['authors'][:3])}"
        
        if len(article['authors']) > 3:
            result += f" et al.\n"
        else:
            result += "\n"
        
        result += f"   Journal: {article['journal']}\n"
        result += f"   Date: {article['pub_date']}\n"
        
        if article['doi'] != "N/A":
            result += f"   DOI: {article['doi']}\n"
        
        result += f"   URL: {article['url']}\n\n"
    
    return result