"""
Drug Information Agent.
Specializes in retrieving drug properties and FDA information.
"""

from crewai import Agent
from tools.pubchem_tools import get_drug_properties
from tools.fda_tools import get_fda_drug_info
import logging

logger = logging.getLogger("pharma_ai.agents.drug_info")

def create_drug_info_agent(llm):
    """
    Create Drug Information Agent.
    
    Args:
        llm: Language model instance
    
    Returns:
        Configured Agent instance
    """
    logger.info("Creating Drug Information Agent")
    
    return Agent(
        role="Drug Information Specialist",
        goal="Retrieve comprehensive drug properties, chemical structures, and FDA regulatory information",
        backstory="""You are a pharmaceutical chemist and regulatory affairs expert with deep 
        knowledge of drug databases including PubChem and FDA resources. You specialize in:
        
        - Analyzing molecular structures and chemical properties
        - Understanding drug mechanisms of action
        - Interpreting FDA labeling and approval information
        - Identifying drug characteristics relevant to repurposing
        - Evaluating pharmacokinetic and pharmacodynamic properties
        
        Your expertise in drug chemistry and regulations helps identify drugs with properties 
        that make them suitable candidates for new therapeutic applications. You can quickly 
        assess whether a drug's chemical profile aligns with potential new indications.""",
        tools=[
            get_drug_properties,
            get_fda_drug_info
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )