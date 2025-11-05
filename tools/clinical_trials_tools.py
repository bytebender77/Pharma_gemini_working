"""
Tools for accessing ClinicalTrials.gov API.
"""

import requests
from typing import Dict, List, Optional
from crewai.tools import tool
import logging
from config import CLINICAL_TRIALS_API, RATE_LIMITS
from utils.api_helpers import rate_limit, retry_on_error
from utils.cache_manager import cache

logger = logging.getLogger("pharma_ai.clinical_trials")

class ClinicalTrialsAPI:
    """Wrapper for ClinicalTrials.gov API v2."""
    
    BASE_URL = CLINICAL_TRIALS_API
    
    @staticmethod
    @rate_limit(RATE_LIMITS["clinical_trials"])
    @retry_on_error(max_attempts=3)
    def search_trials(
        condition: str,
        max_results: int = 20,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Search clinical trials by condition.
        
        Args:
            condition: Disease or condition name
            max_results: Maximum number of results
            status: Trial status (e.g., 'RECRUITING', 'COMPLETED')
        
        Returns:
            List of trial summaries
        """
        # Check cache
        cache_key = f"{condition}_{max_results}_{status}"
        cached_data = cache.get("clinical_trials", cache_key)
        if cached_data:
            return cached_data
        
        params = {
            "query.cond": condition,
            "pageSize": min(max_results, 100),
            "format": "json"
        }
        
        if status:
            params["filter.overallStatus"] = status
        
        try:
            logger.info(f"Searching clinical trials for condition: {condition}")
            response = requests.get(
                ClinicalTrialsAPI.BASE_URL,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            trials = []
            for study in data.get("studies", []):
                protocol = study.get("protocolSection", {})
                identification = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                description = protocol.get("descriptionModule", {})
                conditions_module = protocol.get("conditionsModule", {})
                interventions_module = protocol.get("armsInterventionsModule", {})
                
                trial_data = {
                    "nct_id": identification.get("nctId", "N/A"),
                    "title": identification.get("briefTitle", "N/A"),
                    "status": status_module.get("overallStatus", "N/A"),
                    "phase": status_module.get("expandedAccessInfo", {}).get("hasExpandedAccess", "N/A") or "N/A",
                    "conditions": conditions_module.get("conditions", []),
                    "interventions": [
                        i.get("name", "N/A") 
                        for i in interventions_module.get("interventions", [])
                    ],
                    "brief_summary": description.get("briefSummary", "N/A")[:500],
                    "start_date": status_module.get("startDateStruct", {}).get("date", "N/A"),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date", "N/A"),
                    "enrollment": status_module.get("enrollmentInfo", {}).get("count", "N/A"),
                    "url": f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}"
                }
                
                trials.append(trial_data)
            
            logger.info(f"Found {len(trials)} clinical trials")
            
            # Cache results
            cache.set("clinical_trials", cache_key, trials)
            
            return trials
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching clinical trials: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in clinical trials search: {e}")
            return []
    
    @staticmethod
    @rate_limit(RATE_LIMITS["clinical_trials"])
    @retry_on_error(max_attempts=3)
    def get_trial_by_drug(drug_name: str, max_results: int = 20) -> List[Dict]:
        """
        Search trials by drug/intervention name.
        
        Args:
            drug_name: Name of drug or intervention
            max_results: Maximum number of results
        
        Returns:
            List of trials
        """
        # Check cache
        cache_key = f"drug_{drug_name}_{max_results}"
        cached_data = cache.get("clinical_trials", cache_key)
        if cached_data:
            return cached_data
        
        params = {
            "query.intr": drug_name,
            "pageSize": min(max_results, 100),
            "format": "json"
        }
        
        try:
            logger.info(f"Searching trials for drug: {drug_name}")
            response = requests.get(
                ClinicalTrialsAPI.BASE_URL,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            trials = ClinicalTrialsAPI._parse_trials(data)
            
            logger.info(f"Found {len(trials)} trials for {drug_name}")
            
            # Cache results
            cache.set("clinical_trials", cache_key, trials)
            
            return trials
            
        except Exception as e:
            logger.error(f"Error fetching trials for drug {drug_name}: {e}")
            return []
    
    @staticmethod
    def _parse_trials(data: Dict) -> List[Dict]:
        """Parse trial data from API response."""
        trials = []
        
        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            
            trials.append({
                "nct_id": identification.get("nctId", "N/A"),
                "title": identification.get("briefTitle", "N/A"),
                "status": status_module.get("overallStatus", "N/A"),
                "phase": status_module.get("phase", "N/A"),
                "conditions": conditions_module.get("conditions", []),
                "url": f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}"
            })
        
        return trials


# LangChain Tool Wrappers
@tool
def search_clinical_trials_by_condition(condition: str) -> str:
    """
    Search for clinical trials by disease condition.
    Use this when you need to find clinical trials for a specific disease or medical condition.
    
    Args:
        condition: Disease or condition name (e.g., 'diabetes', 'asthma', 'breast cancer')
    
    Returns:
        Formatted string with trial information including NCT ID, title, status, phase, and interventions
    """
    trials = ClinicalTrialsAPI.search_trials(condition, max_results=10)
    
    if not trials:
        return f"No clinical trials found for condition: {condition}"
    
    result = f"Found {len(trials)} clinical trials for {condition}:\n\n"
    
    for i, trial in enumerate(trials, 1):
        result += f"{i}. {trial['title']}\n"
        result += f"   NCT ID: {trial['nct_id']}\n"
        result += f"   Status: {trial['status']}\n"
        result += f"   Phase: {trial['phase']}\n"
        result += f"   Conditions: {', '.join(trial['conditions'][:3])}\n"
        result += f"   Interventions: {', '.join(trial['interventions'][:3])}\n"
        result += f"   URL: {trial['url']}\n\n"
    
    return result


@tool
def search_trials_by_drug(drug_name: str) -> str:
    """
    Search for clinical trials testing a specific drug or intervention.
    Use this when you need to find trials that are testing a particular medication.
    
    Args:
        drug_name: Name of the drug or intervention (e.g., 'metformin', 'aspirin', 'pembrolizumab')
    
    Returns:
        Formatted string with trial information including trial title, NCT ID, status, and conditions
    """
    trials = ClinicalTrialsAPI.get_trial_by_drug(drug_name, max_results=10)
    
    if not trials:
        return f"No clinical trials found for drug: {drug_name}"
    
    result = f"Found {len(trials)} trials testing {drug_name}:\n\n"
    
    for i, trial in enumerate(trials, 1):
        result += f"{i}. {trial['title']}\n"
        result += f"   NCT ID: {trial['nct_id']}\n"
        result += f"   Status: {trial['status']}\n"
        result += f"   Phase: {trial['phase']}\n"
        result += f"   Conditions: {', '.join(trial.get('conditions', [])[:3])}\n"
        result += f"   URL: {trial['url']}\n\n"
    
    return result