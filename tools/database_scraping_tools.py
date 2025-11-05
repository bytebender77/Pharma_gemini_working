"""Specialized database scraping for rare disease research."""
import requests
from bs4 import BeautifulSoup
from crewai.tools import tool
import logging
from typing import List, Dict
import time

logger = logging.getLogger(__name__)


class RareDiseaseDatabase:
    """Scrape rare disease-specific databases."""
    
    @staticmethod
    def scrape_orphanet(disease_name: str) -> Dict:
        """Scrape Orphanet for rare disease information."""
        try:
            search_url = f"https://www.orpha.net/consor/cgi-bin/Disease_Search.php?lng=EN&data_id=&Disease_Disease_Search_diseaseGroup={disease_name}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract disease info
            disease_info = {
                "name": disease_name,
                "prevalence": "Unknown",
                "inheritance": "Unknown",
                "age_of_onset": "Unknown",
                "orphanet_url": search_url
            }
            
            # Try to extract prevalence
            prev_elem = soup.find(string=lambda text: text and "Prevalence" in text)
            if prev_elem:
                parent = prev_elem.find_parent()
                if parent:
                    disease_info["prevalence"] = parent.get_text(strip=True)
            
            return disease_info
        except Exception as e:
            logger.error(f"Error scraping Orphanet: {e}")
            return {"name": disease_name, "error": str(e)}
    
    @staticmethod
    def scrape_rare_diseases_info(disease_name: str) -> Dict:
        """Scrape rarediseases.info.nih.gov (GARD)."""
        try:
            # Search GARD
            search_url = f"https://rarediseases.info.nih.gov/diseases/{disease_name.replace(' ', '-').lower()}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=15)
            
            if response.status_code == 404:
                # Try search instead
                search_api = f"https://rarediseases.info.nih.gov/api/gard/search?q={disease_name}"
                api_response = requests.get(search_api, headers=headers, timeout=15)
                if api_response.status_code == 200:
                    data = api_response.json()
                    if data and len(data) > 0:
                        first_result = data[0]
                        return {
                            "name": first_result.get('name', disease_name),
                            "gard_id": first_result.get('gard_id', 'N/A'),
                            "synonyms": first_result.get('synonyms', []),
                            "url": f"https://rarediseases.info.nih.gov/diseases/{first_result.get('gard_id', '')}"
                        }
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            return {
                "name": disease_name,
                "content": soup.get_text()[:1000],
                "url": search_url
            }
        except Exception as e:
            logger.error(f"Error scraping GARD: {e}")
            return {"name": disease_name, "error": str(e)}


@tool
def search_orphanet_rare_disease(disease_name: str) -> str:
    """
    Search Orphanet database for rare disease information.
    Use this to get epidemiology, genetics, and clinical data.
    
    Args:
        disease_name: Name of rare disease
    
    Returns:
        Disease prevalence, inheritance, and classification
    """
    data = RareDiseaseDatabase.scrape_orphanet(disease_name)
    
    if "error" in data:
        return f"Could not find {disease_name} in Orphanet: {data['error']}"
    
    result = f"Orphanet Data for {disease_name}:\n\n"
    result += f"Prevalence: {data['prevalence']}\n"
    result += f"Inheritance: {data['inheritance']}\n"
    result += f"Age of Onset: {data['age_of_onset']}\n"
    result += f"Source: {data['orphanet_url']}\n"
    
    return result


@tool
def search_gard_rare_disease(disease_name: str) -> str:
    """
    Search GARD (Genetic and Rare Diseases Information Center).
    Use this for patient-friendly information and resources.
    
    Args:
        disease_name: Name of rare disease
    
    Returns:
        Disease information and resources
    """
    data = RareDiseaseDatabase.scrape_rare_diseases_info(disease_name)
    
    if "error" in data:
        return f"Could not find {disease_name} in GARD"
    
    result = f"GARD Information for {disease_name}:\n\n"
    if "gard_id" in data:
        result += f"GARD ID: {data['gard_id']}\n"
        result += f"Synonyms: {', '.join(data.get('synonyms', []))}\n"
    result += f"URL: {data['url']}\n"
    
    return result