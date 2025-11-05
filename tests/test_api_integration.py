"""
Test API integrations with real endpoints.
Run this to verify all API connections are working.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.clinical_trials_tools import ClinicalTrialsAPI
from tools.pubchem_tools import PubChemAPI
from tools.pubmed_tools import PubMedAPI
from tools.fda_tools import OpenFDAAPI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_clinical_trials():
    """Test ClinicalTrials.gov API."""
    print("\n" + "="*60)
    print("🧪 TESTING CLINICAL TRIALS API")
    print("="*60)
    
    try:
        trials = ClinicalTrialsAPI.search_trials("diabetes", max_results=5)
        
        assert len(trials) > 0, "No trials found"
        assert 'nct_id' in trials[0], "Missing NCT ID"
        assert 'title' in trials[0], "Missing title"
        
        print(f"✅ SUCCESS: Found {len(trials)} trials")
        print(f"   First trial: {trials[0]['title'][:80]}...")
        print(f"   NCT ID: {trials[0]['nct_id']}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_pubchem():
    """Test PubChem API."""
    print("\n" + "="*60)
    print("🧪 TESTING PUBCHEM API")
    print("="*60)
    
    try:
        compound = PubChemAPI.get_compound_by_name("aspirin")
        
        assert compound is not None, "Compound not found"
        assert 'cid' in compound, "Missing CID"
        assert 'molecular_formula' in compound, "Missing molecular formula"
        
        print(f"✅ SUCCESS: Retrieved compound data")
        print(f"   CID: {compound['cid']}")
        print(f"   Formula: {compound['molecular_formula']}")
        print(f"   Weight: {compound['molecular_weight']} g/mol")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_pubmed():
    """Test PubMed API."""
    print("\n" + "="*60)
    print("🧪 TESTING PUBMED API")
    print("="*60)
    
    try:
        pmids = PubMedAPI.search_articles("metformin diabetes", max_results=5)
        
        assert len(pmids) > 0, "No articles found"
        
        print(f"✅ SUCCESS: Found {len(pmids)} articles")
        print(f"   PMIDs: {', '.join(pmids[:3])}...")
        
        # Test article details
        articles = PubMedAPI.fetch_article_details(pmids[:2])
        
        assert len(articles) > 0, "No article details retrieved"
        assert 'title' in articles[0], "Missing title"
        
        print(f"   First article: {articles[0]['title'][:80]}...")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_fda():
    """Test OpenFDA API."""
    print("\n" + "="*60)
    print("🧪 TESTING OPENFDA API")
    print("="*60)
    
    try:
        labels = OpenFDAAPI.search_drug_labels("aspirin", limit=2)
        
        assert len(labels) > 0, "No labels found"
        assert 'brand_name' in labels[0], "Missing brand name"
        assert 'generic_name' in labels[0], "Missing generic name"
        
        print(f"✅ SUCCESS: Found {len(labels)} drug labels")
        print(f"   Brand: {labels[0]['brand_name']}")
        print(f"   Generic: {labels[0]['generic_name']}")
        print(f"   Manufacturer: {labels[0]['manufacturer']}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def run_all_tests():
    """Run all API integration tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "API INTEGRATION TEST SUITE" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {
        "Clinical Trials API": test_clinical_trials(),
        "PubChem API": test_pubchem(),
        "PubMed API": test_pubmed(),
        "OpenFDA API": test_fda()
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    print()


if __name__ == "__main__":
    run_all_tests()