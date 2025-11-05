"""
Setup script for Pharma Intelligence AI
Run this to verify installation and configuration
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        ('streamlit', 'streamlit'),
        ('crewai', 'crewai'),
        ('openai', 'openai'),
        ('tenacity', 'tenacity'),
        ('dotenv', 'dotenv'),
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} not installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_env_file():
    """Check if .env file exists."""
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check for required keys
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv('OPENAI_API_KEY'):
            print("✅ OPENAI_API_KEY is set")
            return True
        else:
            print("⚠️  OPENAI_API_KEY not set in .env")
            return False
    else:
        print("⚠️  .env file not found")
        print("Create .env file with OPENAI_API_KEY")
        return False

def check_directories():
    """Check if required directories exist."""
    directories = ['data/cache', 'outputs', 'data/mock_data']
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"✅ Directory exists: {directory}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    return True

def main():
    """Run setup checks."""
    print("🧬 Pharma Intelligence AI - Setup Check\n")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_env_file),
        ("Directories", check_directories),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        results.append(check_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All checks passed! Ready to deploy.")
        print("\nRun the app with: streamlit run app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

