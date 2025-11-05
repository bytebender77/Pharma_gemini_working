"""Deep research agent for comprehensive analysis."""
from crewai import Agent
from tools.web_scraping_tools import (
    search_google_scholar_papers,
    get_drug_mechanism_details,
    search_biorxiv_preprints,
    extract_webpage_content
)
from tools.database_scraping_tools import (
    search_orphanet_rare_disease,
    search_gard_rare_disease
)


def create_deep_research_agent(llm):
    """Create Deep Research Agent with web scraping capabilities."""
    return Agent(
        role="Deep Research Analyst",
        goal="""Conduct thorough, multi-source research by scraping academic databases, 
        rare disease registries, and pharmaceutical literature to provide comprehensive 
        analysis with detailed mechanisms of action""",
        backstory="""You are a PhD-level pharmaceutical researcher with 20+ years 
        experience in drug discovery and rare disease research. You excel at:
        - Finding obscure research papers and preprints
        - Understanding molecular mechanisms
        - Connecting dots across multiple data sources
        - Identifying novel repurposing opportunities based on mechanism analysis
        - Thoroughly investigating rare disease pathophysiology
        
        You ALWAYS dig deeper than surface-level information. You scrape multiple sources,
        read full papers, analyze mechanisms, and provide evidence-based insights.""",
        tools=[
            search_google_scholar_papers,
            get_drug_mechanism_details,
            search_biorxiv_preprints,
            search_orphanet_rare_disease,
            search_gard_rare_disease,
            extract_webpage_content
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15  # Allow more iterations for thorough research
    )