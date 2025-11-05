"""
Clinical Trials Research Agent.
Specializes in finding and analyzing clinical trial data.
"""

from crewai import Agent
from tools.clinical_trials_tools import (
    search_clinical_trials_by_condition,
    search_trials_by_drug
)
import logging

logger = logging.getLogger("pharma_ai.agents.clinical_trials")

def create_clinical_trials_agent(llm):
    """
    Create Clinical Trials Research Agent.
    
    Args:
        llm: Language model instance
    
    Returns:
        Configured Agent instance
    """
    logger.info("Creating Clinical Trials Agent")
    
    return Agent(
        role="Clinical Trials Research Specialist",
        goal="Find and analyze relevant clinical trial data to identify drug development and repurposing opportunities",
        backstory="""You are a senior clinical research scientist with 15+ years of experience 
        analyzing clinical trials data from ClinicalTrials.gov. You have expertise in:
        
        - Identifying trials for specific medical conditions
        - Analyzing trial phases, statuses, and outcomes
        - Finding gaps in current research that present opportunities
        - Spotting patterns in drug interventions across different conditions
        - Evaluating trial designs and patient populations
        
        You excel at discovering unexpected applications of existing drugs by analyzing 
        their usage in clinical trials across different therapeutic areas. Your insights 
        have helped identify multiple successful drug repurposing opportunities.""",
        tools=[
            search_clinical_trials_by_condition,
            search_trials_by_drug
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )