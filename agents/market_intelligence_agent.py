"""Market Intelligence Agent with real data scraping."""
from crewai import Agent
from tools.market_tools import get_market_data, analyze_competition
from tools.web_scraping_tools import scrape_company_pipeline_info, extract_webpage_content


def create_market_intelligence_agent(llm):
    """Create Market Intelligence Agent with web scraping."""
    return Agent(
        role="Pharmaceutical Market Intelligence Director",
        goal="""Provide comprehensive competitive intelligence by analyzing company 
        pipelines, market trends, pricing strategies, and commercial opportunities 
        using web scraping and market databases""",
        backstory="""You are a former Big Pharma business development director with 
        15+ years in market analysis and competitive intelligence. You excel at:
        - Scraping and analyzing competitor pipelines
        - Assessing market size and growth potential
        - Identifying underserved patient populations
        - Evaluating commercial viability
        - Strategic pricing and reimbursement analysis
        
        You provide REAL competitive intelligence, not generic market overviews.""",
        tools=[
            get_market_data,
            analyze_competition,
            scrape_company_pipeline_info,
            extract_webpage_content
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=10
    )