"""Advanced web scraping tools for pharma research."""
import requests
from bs4 import BeautifulSoup
from crewai.tools import tool
import logging
from typing import Dict, List, Optional
import time
from urllib.parse import quote_plus
import trafilatura
from scholarly import scholarly
import json

logger = logging.getLogger(__name__)


class WebScraperAPI:
    """Advanced web scraping for pharma data."""
    
    @staticmethod
    def scrape_clean_text(url: str) -> str:
        """Extract clean main content from any webpage."""
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            return text or "Could not extract content"
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return f"Error: {e}"
    
    @staticmethod
    def scrape_google_scholar(query: str, max_results: int = 10) -> List[Dict]:
        """Scrape Google Scholar for academic papers."""
        try:
            search_query = scholarly.search_pubs(query)
            results = []
            
            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break
                
                try:
                    filled = scholarly.fill(pub)
                    results.append({
                        "title": filled.get('bib', {}).get('title', 'N/A'),
                        "authors": filled.get('bib', {}).get('author', []),
                        "year": filled.get('bib', {}).get('pub_year', 'N/A'),
                        "venue": filled.get('bib', {}).get('venue', 'N/A'),
                        "citations": filled.get('num_citations', 0),
                        "url": filled.get('pub_url', 'N/A'),
                        "abstract": filled.get('bib', {}).get('abstract', 'N/A')[:500]
                    })
                    time.sleep(2)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error filling publication: {e}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error scraping Google Scholar: {e}")
            return []
    
    @staticmethod
    def scrape_drugbank_info(drug_name: str) -> Optional[Dict]:
        """Scrape DrugBank for detailed drug information."""
        url = f"https://go.drugbank.com/drugs/{drug_name.lower().replace(' ', '_')}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {
                "drug_name": drug_name,
                "drugbank_url": url,
                "mechanism": "",
                "pharmacodynamics": "",
                "indication": "",
                "pharmacokinetics": ""
            }
            
            # Extract mechanism
            mech_section = soup.find('dt', string='Mechanism of action')
            if mech_section:
                dd = mech_section.find_next_sibling('dd')
                if dd:
                    data["mechanism"] = dd.get_text(strip=True)[:500]
            
            # Extract pharmacodynamics
            pharma_section = soup.find('dt', string='Pharmacodynamics')
            if pharma_section:
                dd = pharma_section.find_next_sibling('dd')
                if dd:
                    data["pharmacodynamics"] = dd.get_text(strip=True)[:500]
            
            return data
        except Exception as e:
            logger.error(f"Error scraping DrugBank for {drug_name}: {e}")
            return None
    
    @staticmethod
    def scrape_bioRxiv_preprints(query: str, max_results: int = 5) -> List[Dict]:
        """Scrape bioRxiv for latest preprints."""
        try:
            search_url = f"https://www.biorxiv.org/search/{quote_plus(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            articles = soup.find_all('div', class_='highwire-article-citation', limit=max_results)
            
            for article in articles:
                title_elem = article.find('span', class_='highwire-cite-title')
                author_elem = article.find('span', class_='highwire-citation-authors')
                date_elem = article.find('span', class_='highwire-cite-metadata-date')
                doi_elem = article.find('span', class_='highwire-cite-metadata-doi')
                
                results.append({
                    "title": title_elem.get_text(strip=True) if title_elem else "N/A",
                    "authors": author_elem.get_text(strip=True) if author_elem else "N/A",
                    "date": date_elem.get_text(strip=True) if date_elem else "N/A",
                    "doi": doi_elem.get_text(strip=True) if doi_elem else "N/A",
                    "url": f"https://www.biorxiv.org{article.find('a')['href']}" if article.find('a') else "N/A"
                })
            
            return results
        except Exception as e:
            logger.error(f"Error scraping bioRxiv: {e}")
            return []
    
    @staticmethod
    def scrape_company_pipeline(company_name: str) -> Dict:
        """Scrape pharmaceutical company pipeline information."""
        try:
            # Search for company pipeline page
            search_query = f"{company_name} pharmaceutical pipeline"
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get first result
            first_result = soup.find('a', class_='result__url')
            if first_result:
                pipeline_url = first_result.get('href', '')
                
                # Scrape pipeline page
                if pipeline_url:
                    pipeline_resp = requests.get(pipeline_url, headers=headers, timeout=10)
                    content = trafilatura.extract(pipeline_resp.content, include_tables=True)
                    
                    return {
                        "company": company_name,
                        "url": pipeline_url,
                        "content": content[:2000] if content else "No content extracted"
                    }
            
            return {"company": company_name, "error": "No pipeline page found"}
        except Exception as e:
            logger.error(f"Error scraping company pipeline: {e}")
            return {"company": company_name, "error": str(e)}


@tool
def search_google_scholar_papers(query: str) -> str:
    """
    Search Google Scholar for academic research papers.
    Use this for in-depth literature review and citation analysis.
    
    Args:
        query: Search query (e.g., 'aspirin rare disease repurposing')
    
    Returns:
        Formatted string with paper details including citations
    """
    papers = WebScraperAPI.scrape_google_scholar(query, max_results=8)
    
    if not papers:
        return f"No papers found for query: {query}"
    
    result = f"Google Scholar Results for '{query}':\n\n"
    for i, paper in enumerate(papers, 1):
        result += f"{i}. {paper['title']}\n"
        result += f"   Authors: {', '.join(paper['authors'][:3])}\n"
        result += f"   Year: {paper['year']}\n"
        result += f"   Venue: {paper['venue']}\n"
        result += f"   Citations: {paper['citations']}\n"
        result += f"   Abstract: {paper['abstract']}...\n"
        result += f"   URL: {paper['url']}\n\n"
    
    return result


@tool
def get_drug_mechanism_details(drug_name: str) -> str:
    """
    Get detailed mechanism of action from DrugBank.
    Use this to understand HOW a drug works at molecular level.
    
    Args:
        drug_name: Name of the drug (e.g., 'aspirin', 'metformin')
    
    Returns:
        Detailed mechanism of action and pharmacodynamics
    """
    data = WebScraperAPI.scrape_drugbank_info(drug_name)
    
    if not data:
        return f"Could not retrieve DrugBank data for: {drug_name}"
    
    result = f"DrugBank Information for {drug_name}:\n\n"
    result += f"Mechanism of Action:\n{data['mechanism']}\n\n"
    result += f"Pharmacodynamics:\n{data['pharmacodynamics']}\n\n"
    result += f"Source: {data['drugbank_url']}\n"
    
    return result


@tool
def search_biorxiv_preprints(query: str) -> str:
    """
    Search bioRxiv for latest preprint research papers.
    Use this to find cutting-edge, unpublished research.
    
    Args:
        query: Search query
    
    Returns:
        Latest preprint papers with DOI and dates
    """
    preprints = WebScraperAPI.scrape_bioRxiv_preprints(query, max_results=5)
    
    if not preprints:
        return f"No preprints found for: {query}"
    
    result = f"Latest bioRxiv Preprints for '{query}':\n\n"
    for i, paper in enumerate(preprints, 1):
        result += f"{i}. {paper['title']}\n"
        result += f"   Authors: {paper['authors']}\n"
        result += f"   Date: {paper['date']}\n"
        result += f"   DOI: {paper['doi']}\n"
        result += f"   URL: {paper['url']}\n\n"
    
    return result


@tool
def scrape_company_pipeline_info(company_name: str) -> str:
    """
    Scrape pharmaceutical company pipeline and R&D information.
    Use this for competitive intelligence and market analysis.
    
    Args:
        company_name: Pharmaceutical company name (e.g., 'Pfizer', 'Novartis')
    
    Returns:
        Company pipeline information
    """
    data = WebScraperAPI.scrape_company_pipeline(company_name)
    
    if "error" in data:
        return f"Error retrieving pipeline for {company_name}: {data['error']}"
    
    result = f"Pipeline Information for {company_name}:\n\n"
    result += f"Source: {data['url']}\n\n"
    result += f"Content:\n{data['content']}\n"
    
    return result


@tool
def extract_webpage_content(url: str) -> str:
    """
    Extract clean text content from any webpage.
    Use this to analyze specific articles, press releases, or reports.
    
    Args:
        url: Full URL of the webpage
    
    Returns:
        Extracted clean text content
    """
    content = WebScraperAPI.scrape_clean_text(url)
    return f"Content from {url}:\n\n{content[:3000]}"