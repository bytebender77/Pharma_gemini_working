"""
Pharma Intelligence AI - Enhanced Multi-Stage Research System
"""

import streamlit as st
from dotenv import load_dotenv
import os
import sys
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.master_agent import create_master_agent, run_comprehensive_research_crew
from agents.clinical_trials_agent import create_clinical_trials_agent
from agents.drug_info_agent import create_drug_info_agent
from agents.literature_agent import create_literature_agent
from agents.market_intelligence_agent import create_market_intelligence_agent
from agents.deep_research_agent import create_deep_research_agent
from config import OPENAI_API_KEY, GOOGLE_API_KEY, APP_TITLE, APP_ICON, DEFAULT_OPENAI_MODEL, DEFAULT_GEMINI_MODEL

load_dotenv()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-indicator {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'research_history' not in st.session_state:
    st.session_state.research_history = []


def initialize_llm():
    """Initialize LLM for CrewAI (prefers selected provider, falls back by availability)."""
    provider = st.session_state.get('llm_provider')

    # If both present and no selection yet, default to Gemini for free testing
    if not provider:
        if GOOGLE_API_KEY:
            provider = 'gemini'
        elif OPENAI_API_KEY:
            provider = 'openai'
        else:
            provider = None
        st.session_state['llm_provider'] = provider

    if provider == 'gemini' and GOOGLE_API_KEY:
        try:
            os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
            return DEFAULT_GEMINI_MODEL
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")
            st.stop()
    elif provider == 'openai' and OPENAI_API_KEY:
        try:
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            return DEFAULT_OPENAI_MODEL
        except Exception as e:
            st.error(f"Error initializing OpenAI: {e}")
            st.stop()
    else:
        st.error("❌ No valid LLM API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in .env/Secrets.")
        st.stop()


def main():
    """Main application with multi-stage research."""
    
    st.markdown(f'<h1 class="main-header">{APP_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Multi-Stage AI Research for Drug Repurposing</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if GOOGLE_API_KEY and OPENAI_API_KEY:
            options = ["Gemini (free)", "OpenAI"]
            default_index = 0 if st.session_state.get('llm_provider', 'gemini') == 'gemini' else 1
            selection = st.selectbox(
                "LLM Provider:",
                options=options,
                index=default_index
            )
            # Normalize selection to label (defensive in case an index is returned)
            selection_label = options[selection] if isinstance(selection, int) else str(selection)
            st.session_state['llm_provider'] = 'gemini' if selection_label.lower().startswith("gemini") else 'openai'
            st.success(f"✅ Connected: {selection_label}")
        elif GOOGLE_API_KEY:
            st.session_state['llm_provider'] = 'gemini'
            st.success("✅ Gemini API Connected")
        elif OPENAI_API_KEY:
            st.session_state['llm_provider'] = 'openai'
            st.success("✅ OpenAI API Connected")
        else:
            st.error("❌ API Key Missing")
        
        st.markdown("---")
        
        st.subheader("🔬 Research Depth")
        research_mode = st.select_slider(
            "Select research thoroughness:",
            options=["Quick (5 min)", "Standard (15 min)", "Deep (25-30 min)"],
            value="Standard (15 min)"
        )
        
        st.markdown("---")
        
        st.subheader("📋 Example Queries")
        example_queries = [
            "Find respiratory drugs with potential for lymphangioleiomyomatosis (LAM)",
            "Analyze anti-inflammatory drugs for idiopathic pulmonary fibrosis",
            "Identify COPD drugs that could treat alpha-1 antitrypsin deficiency",
            "Search for asthma medications with potential for primary ciliary dyskinesia"
        ]
        
        selected_example = st.selectbox(
            "Quick start:",
            ["-- Select Example --"] + example_queries,
            index=0
        )
        
        st.markdown("---")
        
        st.subheader("ℹ️ About")
        st.info("""
        **6 AI Agents:**
        
        🎯 Research Director  
        🔬 Clinical Trials Specialist  
        💊 Drug Information Expert  
        📚 Literature Analyst  
        📊 Market Intelligence  
        🧬 Deep Research Analyst
        
        **Data Sources:**
        - ClinicalTrials.gov
        - PubChem & DrugBank
        - PubMed & Google Scholar
        - bioRxiv Preprints
        - Orphanet & GARD
        - Company Pipelines
        """)

        st.markdown("---")
        st.subheader("⚡ Run Options")
        fast_mode = st.checkbox("Skip deep scraping (fast mode)", value=True, help="Skips heavy Stage 2 scraping for faster runs")
        st.session_state['skip_deep'] = fast_mode
    
    # Main content
    st.subheader("🔬 Research Query")
    
    default_query = selected_example if selected_example != "-- Select Example --" else ""
    
    user_query = st.text_area(
        "What would you like to research?",
        value=default_query,
        height=120,
        placeholder="Example: Identify respiratory drugs approved for asthma that could be repurposed for treating lymphangioleiomyomatosis based on shared pathophysiology",
        help="Be specific about target rare disease for best results"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_button = st.button("🚀 Start Deep Research", type="primary")
    
    with col2:
        if st.button("🗑️ Clear"):
            st.rerun()
    
    if analyze_button and user_query:
        st.markdown("---")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        stage_indicator = st.empty()
        
        # Estimated time
        time_estimates = {
            "Quick (5 min)": 5,
            "Standard (15 min)": 15,
            "Deep (25-30 min)": 27
        }
        estimated_time = time_estimates.get(research_mode, 15)
        
        status_text.info(f"🤖 **Multi-Stage Research Initiated** (Est. {estimated_time} min)")
        
        start_time = time.time()
        
        try:
            # Initialize LLM
            llm = initialize_llm()
            
            # Create all agents
            with stage_indicator.container():
                st.markdown('<div class="stage-indicator">📦 <b>Initializing AI Agents...</b></div>', unsafe_allow_html=True)
            
            progress_bar.progress(10)
            
            clinical_agent = create_clinical_trials_agent(llm)
            drug_info_agent = create_drug_info_agent(llm)
            literature_agent = create_literature_agent(llm)
            market_agent = create_market_intelligence_agent(llm)
            deep_research_agent = create_deep_research_agent(llm)
            master_agent = create_master_agent(llm)
            
            progress_bar.progress(20)
            
            # Run multi-stage research
            with stage_indicator.container():
                st.markdown('''
                <div class="stage-indicator">
                    <b>🔍 STAGE 1: Initial Discovery</b><br>
                    Searching clinical trials, basic literature, candidate drugs...
                </div>
                ''', unsafe_allow_html=True)
            
            progress_bar.progress(30)
            
            result = run_comprehensive_research_crew(
                user_query=user_query,
                master_agent=master_agent,
                clinical_agent=clinical_agent,
                drug_agent=drug_info_agent,
                literature_agent=literature_agent,
                market_agent=market_agent,
                deep_research_agent=deep_research_agent,
                llm=llm,
                skip_deep=st.session_state.get('skip_deep', True)
            )
            
            progress_bar.progress(100)
            
            elapsed_time = int((time.time() - start_time) / 60)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            stage_indicator.empty()
            
            # Show success
            st.success(f"✅ **Research Complete!** (Completed in {elapsed_time} minutes)")
            
            # Display results
            st.markdown("### 📄 Comprehensive Research Report")
            st.markdown(result)
            
            # Save to history
            st.session_state.research_history.append({
                "query": user_query,
                "result": result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration": f"{elapsed_time} min"
            })
            
            # Download
            st.markdown("---")
            st.download_button(
                label="📥 Download Full Report (TXT)",
                data=f"PHARMA INTELLIGENCE AI - RESEARCH REPORT\nGenerated: {datetime.now()}\nDuration: {elapsed_time} min\n\nQUERY:\n{user_query}\n\n{result}",
                file_name=f"pharma_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            stage_indicator.empty()
            
            st.error(f"❌ **Error:** {str(e)}")
            with st.expander("🔍 Debug Info"):
                st.exception(e)
    
    elif analyze_button:
        st.warning("⚠️ Please enter a research question")
    
    # History
    if st.session_state.research_history:
        st.markdown("---")
        st.subheader("📚 Research History")
        
        for idx, item in enumerate(reversed(st.session_state.research_history[-5:]), 1):
            with st.expander(f"🔬 {item['query'][:60]}... ({item['duration']})"):
                st.markdown(item['result'])


if __name__ == "__main__":
    main()