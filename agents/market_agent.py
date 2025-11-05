"""
Market Analysis Agent.
Specializes in pharmaceutical market intelligence.
"""

from crewai import Agent
from tools.market_tools import get_market_data, analyze_competition
import logging

logger = logging.getLogger("pharma_ai.agents.market")

def create_market_agent(llm):
    """
    Create Market Analysis Agent.
    
    Args:
        llm: Language model instance
    
    Returns:
        Configured Agent instance
    """
    logger.info("Creating Market Agent")
    
    return Agent(
        role="Pharmaceutical Market Intelligence Analyst",
        goal="Analyze market opportunities, competitive landscapes, and commercial viability of drug repurposing opportunities",
        backstory="""You are a senior pharmaceutical market analyst with an MBA and 12+ years 
        of experience in pharma business intelligence. Your expertise includes:
        
        - Analyzing market size, growth trends, and market dynamics
        - Evaluating competitive landscapes and market positioning
        - Assessing commercial viability of new indications
        - Understanding patent landscapes and exclusivity windows
        - Identifying unmet medical needs with strong market potential
        
        You have successfully guided multiple drug repurposing programs by identifying 
        attractive market opportunities that balance clinical need with commercial potential. 
        Your market insights help prioritize opportunities that are both scientifically 
        sound and commercially viable.""",
        tools=[
            get_market_data,
            analyze_competition
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )