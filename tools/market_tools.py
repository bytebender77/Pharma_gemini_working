"""
Mock tools for market data (simulating IQVIA/market intelligence).
"""

from typing import Dict
from crewai.tools import tool
import random
import logging

logger = logging.getLogger("pharma_ai.market")

# Mock IQVIA-style market data
MOCK_MARKET_DATA = {
    "respiratory": {
        "market_size_usd_m": 85000,
        "cagr_percent": 12.5,
        "competition_level": "Medium",
        "top_players": ["GlaxoSmithKline", "AstraZeneca", "Novartis", "Boehringer Ingelheim"],
        "patient_population_m": 334.0,
        "key_drugs": ["Advair", "Symbicort", "Spiriva", "Dupixent", "Trelegy"],
        "emerging_trends": ["Biologics for severe asthma", "Digital inhalers", "Personalized medicine"],
        "market_share_leader": "GSK (18%)"
    },
    "diabetes": {
        "market_size_usd_m": 210000,
        "cagr_percent": 8.3,
        "competition_level": "High",
        "top_players": ["Novo Nordisk", "Sanofi", "Eli Lilly", "AstraZeneca"],
        "patient_population_m": 537.0,
        "key_drugs": ["Ozempic", "Lantus", "Trulicity", "Jardiance", "Mounjaro"],
        "emerging_trends": ["GLP-1 receptor agonists", "SGLT2 inhibitors", "Smart insulin pens"],
        "market_share_leader": "Novo Nordisk (28%)"
    },
    "cardiovascular": {
        "market_size_usd_m": 150000,
        "cagr_percent": 6.5,
        "competition_level": "High",
        "top_players": ["Pfizer", "Novartis", "Merck", "Bristol Myers Squibb"],
        "patient_population_m": 422.0,
        "key_drugs": ["Eliquis", "Entresto", "Crestor", "Brilinta", "Xarelto"],
        "emerging_trends": ["Novel anticoagulants", "PCSK9 inhibitors", "RNA therapeutics"],
        "market_share_leader": "Pfizer (22%)"
    },
    "oncology": {
        "market_size_usd_m": 320000,
        "cagr_percent": 15.2,
        "competition_level": "Medium",
        "top_players": ["Roche", "Bristol Myers Squibb", "Merck", "AstraZeneca"],
        "patient_population_m": 19.3,
        "key_drugs": ["Keytruda", "Opdivo", "Tecentriq", "Imbruvica", "Revlimid"],
        "emerging_trends": ["Immunotherapy", "CAR-T therapies", "Precision oncology", "Antibody-drug conjugates"],
        "market_share_leader": "Roche (24%)"
    },
    "neurology": {
        "market_size_usd_m": 95000,
        "cagr_percent": 10.8,
        "competition_level": "Medium",
        "top_players": ["Biogen", "Novartis", "Roche", "UCB"],
        "patient_population_m": 156.0,
        "key_drugs": ["Ocrevus", "Copaxone", "Tecfidera", "Spinraza", "Gilenya"],
        "emerging_trends": ["Gene therapy", "Neurodegeneration treatments", "Digital therapeutics"],
        "market_share_leader": "Biogen (19%)"
    },
    "immunology": {
        "market_size_usd_m": 125000,
        "cagr_percent": 13.7,
        "competition_level": "High",
        "top_players": ["AbbVie", "Johnson & Johnson", "Amgen", "Novartis"],
        "patient_population_m": 78.0,
        "key_drugs": ["Humira", "Stelara", "Enbrel", "Cosentyx", "Skyrizi"],
        "emerging_trends": ["IL-23 inhibitors", "JAK inhibitors", "Biosimilars"],
        "market_share_leader": "AbbVie (31%)"
    },
    "rare_diseases": {
        "market_size_usd_m": 58000,
        "cagr_percent": 18.5,
        "competition_level": "Low",
        "top_players": ["Alexion", "BioMarin", "Vertex", "Sarepta"],
        "patient_population_m": 2.5,
        "key_drugs": ["Soliris", "Spinraza", "Trikafta", "Luxturna"],
        "emerging_trends": ["Gene therapy", "RNA therapeutics", "Orphan drug development"],
        "market_share_leader": "Alexion (22%)"
    }
}

@tool
def get_market_data(therapy_area: str) -> str:
    """
    Get market intelligence data for a therapeutic area.
    Use this when you need market size, growth rates, competition analysis, or key players information.
    
    Args:
        therapy_area: Name of therapy area (e.g., 'respiratory', 'diabetes', 'cardiovascular', 
                      'oncology', 'neurology', 'immunology', 'rare_diseases')
    
    Returns:
        Formatted string with comprehensive market analysis including market size, CAGR,
        competition level, top pharmaceutical companies, patient population, and key drugs
    """
    therapy_area_normalized = therapy_area.lower().replace(" ", "_")
    
    # Try to find matching therapy area
    data = None
    for key in MOCK_MARKET_DATA.keys():
        if key in therapy_area_normalized or therapy_area_normalized in key:
            data = MOCK_MARKET_DATA[key]
            therapy_area = key
            break
    
    if not data:
        # Return list of available areas
        available = ", ".join(MOCK_MARKET_DATA.keys())
        return f"Market data not available for: {therapy_area}\nAvailable areas: {available}"
    
    result = f"Market Intelligence - {therapy_area.replace('_', ' ').title()}:\n\n"
    result += f"📊 Market Overview:\n"
    result += f"• Global Market Size: ${data['market_size_usd_m']:,}M USD\n"
    result += f"• Growth Rate (CAGR): {data['cagr_percent']}%\n"
    result += f"• Competition Level: {data['competition_level']}\n"
    result += f"• Market Leader: {data['market_share_leader']}\n\n"
    
    result += f"👥 Patient Demographics:\n"
    result += f"• Global Patient Population: {data['patient_population_m']}M\n\n"
    
    result += f"🏆 Top Market Players:\n"
    for player in data['top_players']:
        result += f"• {player}\n"
    
    result += f"\n💊 Leading Drugs:\n"
    for drug in data['key_drugs']:
        result += f"• {drug}\n"
    
    result += f"\n🔬 Emerging Trends:\n"
    for trend in data['emerging_trends']:
        result += f"• {trend}\n"
    
    return result


@tool
def analyze_competition(drug_name: str) -> str:
    """
    Analyze competitive landscape for a specific drug.
    Use this when you need to understand market competition, positioning, and opportunities.
    
    Args:
        drug_name: Name of the drug to analyze (e.g., 'metformin', 'aspirin', 'keytruda')
    
    Returns:
        Formatted string with competitive analysis including number of competitors,
        market share estimates, market position, and patent status
    """
    # Mock competitive data with realistic variations
    competitors = random.randint(3, 25)
    market_share = round(random.uniform(2.5, 35.0), 1)
    
    # Determine market position
    if market_share > 25:
        position = "Market Leader"
    elif market_share > 15:
        position = "Strong Challenger"
    elif market_share > 8:
        position = "Established Player"
    else:
        position = "Niche Player"
    
    # Patent status
    patent_years_remaining = random.randint(0, 12)
    if patent_years_remaining == 0:
        patent_status = "Patent Expired - Generic Competition Active"
    elif patent_years_remaining < 3:
        patent_status = f"Patent Expiring Soon ({patent_years_remaining} years remaining)"
    else:
        patent_status = f"Patent Protected ({patent_years_remaining} years remaining)"
    
    result = f"Competitive Analysis for {drug_name}:\n\n"
    result += f"🎯 Market Position:\n"
    result += f"• Market Position: {position}\n"
    result += f"• Estimated Market Share: {market_share}%\n"
    result += f"• Number of Direct Competitors: {competitors}\n\n"
    
    result += f"⚖️ Patent & IP Status:\n"
    result += f"• {patent_status}\n\n"
    
    result += f"💡 Strategic Insights:\n"
    
    if market_share > 20:
        result += f"• Strong market position with significant share\n"
        result += f"• Focus on market expansion and line extensions\n"
    else:
        result += f"• Opportunity for market share growth\n"
        result += f"• Consider differentiation strategies\n"
    
    if patent_years_remaining < 3:
        result += f"• Patent cliff approaching - generic competition expected\n"
        result += f"• Explore life cycle management strategies\n"
        result += f"• Consider repositioning or new indications\n"
    
    if competitors > 15:
        result += f"• Highly competitive market\n"
        result += f"• Differentiation and value proposition critical\n"
    else:
        result += f"• Moderate competition level\n"
        result += f"• Opportunity for market penetration\n"
    
    return result