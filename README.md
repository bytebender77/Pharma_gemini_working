# 🧬 Pharma Intelligence AI

Multi-Agent AI System for Drug Repurposing and Market Intelligence

## Features

- 🔬 **Clinical Trials Analysis** - Search ClinicalTrials.gov for relevant trials
- 💊 **Drug Properties** - Retrieve comprehensive drug information from PubChem and FDA
- 📚 **Literature Review** - Search and analyze scientific publications from PubMed
- 📊 **Market Intelligence** - Analyze market opportunities and competitive landscapes
- 🤖 **5 Specialized AI Agents** - Coordinated multi-agent research system
- 📄 **Automated Report Generation** - Comprehensive research reports with citations

## Quick Start

### Local Development

1. **Clone Repository**
```bash
git clone <your-repo-url>
cd Pharma-AI-Agent
```

2. **Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API Keys:
- `OPENAI_API_KEY` - Get from [OpenAI dashboard](https://platform.openai.com/)

Optional API Keys:
- `NCBI_API_KEY` - For higher PubMed rate limits (optional)

5. **Run the Application**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
   - Create a repository on GitHub
   - Push your code to the repository

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `app.py`
   - Add secrets (API keys) in the Streamlit Cloud dashboard:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Configuration**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`
   - Environment variables should be set in Streamlit Cloud's secrets management

### Other Deployment Options

#### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Manual Server Deployment
1. Install Python 3.11+ and dependencies
2. Set environment variables
3. Run: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional
GROQ_API_KEY=your_groq_api_key_here
NCBI_API_KEY=your_ncbi_api_key_here

# Optional Configuration
DEBUG_MODE=false
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
MAX_SEARCH_RESULTS=20
```

### Streamlit Configuration

Streamlit settings are configured in `.streamlit/config.toml`. Modify as needed for your deployment.

## Project Structure

```
pharma-agentic-ai/
├── agents/              # AI agent definitions
│   ├── master_agent.py
│   ├── clinical_trials_agent.py
│   ├── drug_info_agent.py
│   ├── literature_agent.py
│   └── market_agent.py
├── tools/               # API integration tools
│   ├── clinical_trials_tools.py
│   ├── pubchem_tools.py
│   ├── pubmed_tools.py
│   ├── fda_tools.py
│   └── market_tools.py
├── utils/               # Utility functions
│   ├── api_helpers.py
│   ├── cache_manager.py
│   ├── data_processor.py
│   └── logger.py
├── data/                # Data storage (cached API responses)
├── outputs/             # Generated reports
├── app.py               # Main Streamlit application
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── README.md
```

## Usage Examples

### Example Queries

1. **Drug Information**
   - "Get comprehensive information about aspirin"
   - "What are the properties of metformin?"

2. **Clinical Trials**
   - "What are the latest clinical trials for diabetes?"
   - "Find trials testing drugs for Alzheimer's disease"

3. **Drug Repurposing**
   - "Find drugs that could be repurposed for treating obesity"
   - "Analyze metformin for non-diabetes indications"

4. **Market Analysis**
   - "Analyze the diabetes drug market opportunity"
   - "What is the competitive landscape for oncology drugs?"

## API Rate Limits

The application respects rate limits for all APIs:
- **PubMed**: 3 requests/second (10 with API key)
- **ClinicalTrials.gov**: 10 requests/second
- **PubChem**: 5 requests/second
- **FDA**: 10 requests/second
- **Google Gemini**: 15 requests/minute

Caching is enabled by default to reduce API calls.

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `GOOGLE_API_KEY` is set in `.env` or Streamlit secrets
   - Verify the API key is valid and has sufficient quota

2. **Import Errors**
   - Reinstall dependencies: `pip install -r requirements.txt --upgrade`
   - Ensure Python 3.11+ is being used

3. **Rate Limiting**
   - Check API quota limits
   - Enable caching (default: enabled)
   - Reduce `MAX_SEARCH_RESULTS` in config

4. **Memory Issues**
   - Reduce `max_iter` in agent configurations
   - Clear cache: Use the "Clear Cache" button in the app sidebar

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Support

For issues or questions, please open an issue on GitHub.

## Acknowledgments

- **CrewAI** - Multi-agent orchestration framework
- **Google Gemini** - LLM provider
- **ClinicalTrials.gov** - Clinical trials data
- **PubChem** - Chemical compound data
- **PubMed** - Scientific literature database
- **OpenFDA** - FDA drug information
