"""
Tools for accessing PubChem API.
"""

import requests
from typing import Dict, Optional, List
from crewai.tools import tool
import logging
from config import PUBCHEM_API, RATE_LIMITS
from utils.api_helpers import rate_limit, retry_on_error
from utils.cache_manager import cache

logger = logging.getLogger("pharma_ai.pubchem")

class PubChemAPI:
    """Wrapper for PubChem REST API."""
    
    BASE_URL = PUBCHEM_API
    
    @staticmethod
    @rate_limit(RATE_LIMITS["pubchem"])
    @retry_on_error(max_attempts=3)
    def get_compound_by_name(compound_name: str) -> Optional[Dict]:
        """
        Get compound information by name.
        
        Args:
            compound_name: Chemical/drug name
        
        Returns:
            Dictionary with compound properties
        """
        # Check cache
        cached_data = cache.get("pubchem", compound_name)
        if cached_data:
            return cached_data
        
        try:
            logger.info(f"Fetching PubChem data for: {compound_name}")
            
            # Get CID (Compound ID)
            cid_url = f"{PubChemAPI.BASE_URL}/compound/name/{compound_name}/cids/JSON"
            response = requests.get(cid_url, timeout=10)
            response.raise_for_status()
            
            cid_data = response.json()
            if "IdentifierList" not in cid_data or not cid_data["IdentifierList"].get("CID"):
                logger.warning(f"No CID found for {compound_name}")
                return None
            
            cid = cid_data["IdentifierList"]["CID"][0]
            
            # Get compound properties
            props_url = f"{PubChemAPI.BASE_URL}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName,InChI,InChIKey/JSON"
            props_response = requests.get(props_url, timeout=10)
            props_response.raise_for_status()
            
            properties = props_response.json()["PropertyTable"]["Properties"][0]
            
            # Get synonyms
            syn_url = f"{PubChemAPI.BASE_URL}/compound/cid/{cid}/synonyms/JSON"
            syn_response = requests.get(syn_url, timeout=10)
            syn_response.raise_for_status()
            
            synonyms_data = syn_response.json()
            synonyms = synonyms_data["InformationList"]["Information"][0]["Synonym"][:15]
            
            # Get description
            desc_url = f"{PubChemAPI.BASE_URL}/compound/cid/{cid}/description/JSON"
            try:
                desc_response = requests.get(desc_url, timeout=10)
                desc_response.raise_for_status()
                desc_data = desc_response.json()
                description = desc_data.get("InformationList", {}).get("Information", [{}])[0].get("Description", "N/A")
            except:
                description = "N/A"
            
            compound_data = {
                "cid": cid,
                "molecular_formula": properties.get("MolecularFormula", "N/A"),
                "molecular_weight": properties.get("MolecularWeight", "N/A"),
                "iupac_name": properties.get("IUPACName", "N/A"),
                "canonical_smiles": properties.get("CanonicalSMILES", "N/A"),
                "inchi": properties.get("InChI", "N/A"),
                "inchi_key": properties.get("InChIKey", "N/A"),
                "synonyms": synonyms,
                "description": description[:500] if description != "N/A" else "N/A",
                "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            }
            
            logger.info(f"Successfully fetched data for CID: {cid}")
            
            # Cache results
            cache.set("pubchem", compound_name, compound_data)
            
            return compound_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching PubChem data for {compound_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in PubChem API: {e}")
            return None
    
    @staticmethod
    @rate_limit(RATE_LIMITS["pubchem"])
    def search_by_similarity(smiles: str, threshold: int = 90) -> List[int]:
        """
        Search for similar compounds by SMILES structure.
        
        Args:
            smiles: SMILES structure string
            threshold: Similarity threshold (0-100)
        
        Returns:
            List of similar compound CIDs
        """
        try:
            logger.info(f"Searching similar compounds with threshold {threshold}")
            
            url = f"{PubChemAPI.BASE_URL}/compound/fastsimilarity_2d/smiles/{smiles}/cids/JSON"
            params = {"Threshold": threshold}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])[:10]
            
            logger.info(f"Found {len(cids)} similar compounds")
            return cids
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []


@tool
def get_drug_properties(drug_name: str) -> str:
    """
    Get chemical and physical properties of a drug or compound from PubChem.
    Use this when you need molecular information, chemical structure, or identifiers for a drug.
    
    Args:
        drug_name: Name of the drug/compound (e.g., 'aspirin', 'metformin', 'ibuprofen')
    
    Returns:
        Formatted string with comprehensive drug properties including molecular formula, 
        weight, SMILES, synonyms, and PubChem URL
    """
    compound = PubChemAPI.get_compound_by_name(drug_name)
    
    if not compound:
        return f"Could not find compound information for: {drug_name}"
    
    result = f"Drug Properties for {drug_name}:\n\n"
    result += f"• PubChem CID: {compound['cid']}\n"
    result += f"• Molecular Formula: {compound['molecular_formula']}\n"
    result += f"• Molecular Weight: {compound['molecular_weight']} g/mol\n"
    result += f"• IUPAC Name: {compound['iupac_name']}\n"
    result += f"• Canonical SMILES: {compound['canonical_smiles']}\n"
    result += f"• InChI Key: {compound['inchi_key']}\n"
    result += f"• Common Synonyms: {', '.join(compound['synonyms'][:5])}\n"
    
    if compound['description'] != "N/A":
        result += f"• Description: {compound['description']}\n"
    
    result += f"• PubChem URL: {compound['pubchem_url']}\n"
    
    return result