"""
Tools for accessing OpenFDA API.
"""

import requests
from typing import Dict, List, Optional
from crewai.tools import tool
import logging
from config import FDA_API, RATE_LIMITS
from utils.api_helpers import rate_limit, retry_on_error
from utils.cache_manager import cache

logger = logging.getLogger("pharma_ai.fda")

class OpenFDAAPI:
    """Wrapper for OpenFDA API."""
    
    BASE_URL = FDA_API
    
    @staticmethod
    @rate_limit(RATE_LIMITS["fda"])
    @retry_on_error(max_attempts=3)
    def search_drug_labels(drug_name: str, limit: int = 5) -> List[Dict]:
        """
        Search FDA drug labels.
        
        Args:
            drug_name: Drug name
            limit: Maximum results
        
        Returns:
            List of drug label information
        """
        # Check cache
        cache_key = f"{drug_name}_{limit}"
        cached_data = cache.get("fda_labels", cache_key)
        if cached_data:
            return cached_data
        
        url = f"{OpenFDAAPI.BASE_URL}/label.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": limit
        }
        
        try:
            logger.info(f"Searching FDA labels for: {drug_name}")
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for result in data.get("results", []):
                openfda = result.get("openfda", {})
                
                label_data = {
                    "brand_name": openfda.get("brand_name", ["N/A"])[0] if openfda.get("brand_name") else "N/A",
                    "generic_name": openfda.get("generic_name", ["N/A"])[0] if openfda.get("generic_name") else "N/A",
                    "manufacturer": openfda.get("manufacturer_name", ["N/A"])[0] if openfda.get("manufacturer_name") else "N/A",
                    "product_type": openfda.get("product_type", ["N/A"])[0] if openfda.get("product_type") else "N/A",
                    "route": openfda.get("route", []) if openfda.get("route") else [],
                    "substance_name": openfda.get("substance_name", []) if openfda.get("substance_name") else [],
                    "indications": result.get("indications_and_usage", ["N/A"])[0][:800] if result.get("indications_and_usage") else "N/A",
                    "dosage": result.get("dosage_and_administration", ["N/A"])[0][:500] if result.get("dosage_and_administration") else "N/A",
                    "warnings": result.get("warnings", ["N/A"])[0][:500] if result.get("warnings") else "N/A",
                    "adverse_reactions": result.get("adverse_reactions", ["N/A"])[0][:500] if result.get("adverse_reactions") else "N/A"
                }
                
                results.append(label_data)
            
            logger.info(f"Found {len(results)} FDA labels")
            
            # Cache results
            cache.set("fda_labels", cache_key, results)
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching FDA labels: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in FDA API: {e}")
            return []
    
    @staticmethod
    @rate_limit(RATE_LIMITS["fda"])
    def get_drug_events(drug_name: str, limit: int = 10) -> List[Dict]:
        """
        Get adverse event reports for a drug.
        
        Args:
            drug_name: Drug name
            limit: Maximum results
        
        Returns:
            List of adverse events
        """
        url = f"{OpenFDAAPI.BASE_URL}/event.json"
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": limit
        }
        
        try:
            logger.info(f"Fetching adverse events for: {drug_name}")
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            events = data.get("results", [])
            
            logger.info(f"Found {len(events)} adverse events")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching drug events: {e}")
            return []
    
    @staticmethod
    @rate_limit(RATE_LIMITS["fda"])
    def search_drug_approvals(drug_name: str) -> List[Dict]:
        """
        Search FDA drug approval applications.
        
        Args:
            drug_name: Drug name
        
        Returns:
            List of approval information
        """
        url = f"{OpenFDAAPI.BASE_URL}/drugsfda.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            "limit": 5
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
            
        except Exception as e:
            logger.error(f"Error fetching drug approvals: {e}")
            return []


@tool
def get_fda_drug_info(drug_name: str) -> str:
    """
    Get FDA-approved drug information including indications, warnings, and manufacturer details.
    Use this when you need official FDA labeling information for a drug.
    
    Args:
        drug_name: Name of the drug (e.g., 'aspirin', 'lipitor', 'metformin')
    
    Returns:
        Formatted string with comprehensive FDA drug information including brand name,
        generic name, manufacturer, indications, dosage, warnings, and adverse reactions
    """
    labels = OpenFDAAPI.search_drug_labels(drug_name, limit=3)
    
    if not labels:
        return f"No FDA information found for: {drug_name}"
    
    result = f"FDA-Approved Drug Information for {drug_name}:\n\n"
    
    for i, label in enumerate(labels, 1):
        result += f"{i}. {label['brand_name']}\n"
        result += f"   Generic Name: {label['generic_name']}\n"
        result += f"   Manufacturer: {label['manufacturer']}\n"
        result += f"   Product Type: {label['product_type']}\n"
        
        if label['route']:
            result += f"   Routes of Administration: {', '.join(label['route'])}\n"
        
        if label['substance_name']:
            result += f"   Active Substances: {', '.join(label['substance_name'][:3])}\n"
        
        result += f"\n   Indications and Usage:\n   {label['indications'][:400]}...\n"
        
        if label['dosage'] != "N/A":
            result += f"\n   Dosage Information:\n   {label['dosage'][:300]}...\n"
        
        if label['warnings'] != "N/A":
            result += f"\n   Warnings:\n   {label['warnings'][:300]}...\n"
        
        result += "\n" + "="*50 + "\n\n"
    
    return result