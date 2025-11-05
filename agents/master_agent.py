"""Enhanced Master Orchestrator with Multi-Stage Research."""
from crewai import Agent, Task, Crew, Process
from typing import List


def create_master_agent(llm):
    """Create Master Orchestrator Agent."""
    return Agent(
        role="Chief Research Officer",
        goal="""Orchestrate comprehensive, multi-stage pharmaceutical research to identify 
        drug repurposing opportunities with high scientific and commercial potential""",
        backstory="""You are a former pharmaceutical R&D executive who led drug 
        discovery programs at top-tier companies. You understand that QUALITY research 
        takes TIME. You coordinate teams methodically through multiple research stages:
        
        STAGE 1: Initial Discovery (5-10 min)
        - Identify candidate drugs and diseases
        - Search clinical trials and basic literature
        
        STAGE 2: Deep Analysis (10-15 min)
        - Scrape academic databases for mechanisms
        - Analyze rare disease databases
        - Review cutting-edge preprints
        
        STAGE 3: Market Validation (5-10 min)
        - Competitive intelligence gathering
        - Commercial feasibility assessment
        
        STAGE 4: Synthesis & Recommendations (5 min)
        - Integrate all findings
        - Generate strategic recommendations
        
        You NEVER rush research. You ensure thorough, evidence-based analysis.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=20
    )


def run_comprehensive_research_crew(
    user_query: str,
    master_agent: Agent,
    clinical_agent: Agent,
    drug_agent: Agent,
    literature_agent: Agent,
    market_agent: Agent,
    deep_research_agent: Agent,
    llm,
    skip_deep: bool = False
) -> str:
    """Execute multi-stage comprehensive research."""
    
    # STAGE 1: Initial Discovery
    stage1_task = Task(
        description=f"""
        STAGE 1: INITIAL DISCOVERY (Target: 5-10 minutes)
        
        Research Question: "{user_query}"
        
        Objectives:
        1. Identify 5-10 candidate respiratory drugs currently approved for common conditions
        2. Search ClinicalTrials.gov for any trials in rare diseases
        3. Get basic PubMed literature (10-15 papers)
        4. List potential rare respiratory diseases for repurposing
        
        Deliverable: Initial candidate list with brief rationale for each drug
        """,
        agent=master_agent,
        expected_output="List of 5-10 candidate drugs with initial evidence for rare disease potential"
    )
    
    # STAGE 2: Deep Mechanism Analysis
    stage2_task = Task(
        description=f"""
        STAGE 2: DEEP MECHANISM ANALYSIS (Target: 10-15 minutes)
        
        Based on Stage 1 findings, conduct deep analysis:
        
        For TOP 3 candidate drugs:
        1. Scrape DrugBank for detailed mechanism of action
        2. Search Google Scholar for highly-cited mechanistic studies
        3. Find bioRxiv preprints on novel uses
        4. Search Orphanet for target rare diseases
        5. Analyze mechanistic overlap between approved indication and target rare disease
        
        For TOP 3 rare diseases:
        1. Search GARD for pathophysiology
        2. Find latest research on disease mechanisms
        3. Identify molecular targets
        
        Deliverable: Detailed mechanism-based analysis explaining WHY repurposing could work
        """,
        agent=deep_research_agent,
        expected_output="Mechanistic analysis showing molecular basis for repurposing opportunities",
        context=[stage1_task]
    )
    
    # STAGE 3: Market & Competitive Intelligence
    stage3_task = Task(
        description=f"""
        STAGE 3: MARKET VALIDATION (Target: 5-10 minutes)
        
        For top opportunities from Stage 2:
        1. Scrape competitor pipelines (Pfizer, Novartis, Roche, etc.)
        2. Assess market size for rare diseases
        3. Analyze competitive landscape
        4. Evaluate commercial feasibility
        5. Assess orphan drug designation potential
        
        Deliverable: Commercial viability assessment with competitive positioning
        """,
        agent=market_agent,
        expected_output="Market opportunity analysis with competitive intelligence",
        context=[stage1_task] if skip_deep else [stage1_task, stage2_task]
    )
    
    # STAGE 4: Final Synthesis
    stage4_task = Task(
        description=f"""
        STAGE 4: SYNTHESIS & STRATEGIC RECOMMENDATIONS (Target: 5 minutes)
        
        Integrate ALL findings from Stages 1-3 to create comprehensive report:
        
        STRUCTURE:
        1. Executive Summary (3-5 key insights)
        2. Top 3 Repurposing Opportunities (ranked by potential)
           - Drug name & current use
           - Target rare disease
           - Mechanistic rationale (detailed)
           - Clinical evidence
           - Market potential
           - Competitive landscape
           - Risk assessment
        3. Strategic Recommendations
           - Next steps for clinical development
           - Partnership opportunities
           - Regulatory pathway
           - Timeline estimates
        4. Full Appendix
           - All clinical trials found
           - Complete literature citations
           - Market data sources
        
        QUALITY STANDARDS:
        - All claims must be evidence-based
        - Cite specific NCT IDs, PMIDs, papers
        - Explain mechanisms in detail
        - Provide realistic market assessments
        - Acknowledge limitations and risks
        
        Deliverable: Publication-ready research report (5000+ words)
        """,
        agent=master_agent,
        expected_output="Comprehensive strategic research report with detailed recommendations",
        context=[stage1_task, stage3_task] if skip_deep else [stage1_task, stage2_task, stage3_task]
    )
    
    # Create crew with sequential processing
    agents_list = [
        master_agent,
        clinical_agent,
        drug_agent,
        literature_agent,
        market_agent
    ]
    if not skip_deep:
        agents_list.append(deep_research_agent)

    tasks_list = [stage1_task]
    if not skip_deep:
        tasks_list.append(stage2_task)
    tasks_list.append(stage3_task)
    tasks_list.append(stage4_task)

    crew = Crew(
        agents=agents_list,
        tasks=tasks_list,
        process=Process.sequential,  # Execute tasks in order
        verbose=True,
        full_output=True
    )
    
    try:
        result = crew.kickoff()
        
        # Extract final output
        if hasattr(result, 'raw'):
            return str(result.raw)
        elif hasattr(result, 'output'):
            return str(result.output)
        else:
            return str(result)
    except Exception as e:
        return f"Error during research execution: {str(e)}\n\nPlease try a more specific query."