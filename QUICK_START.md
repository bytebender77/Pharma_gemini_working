# Quick Start Guide

## 🚀 Deploy in 5 Minutes

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect GitHub repo
   - Main file: `app.py`
   - Add secret: `GOOGLE_API_KEY`

3. **Done!** Your app will be live in ~2 minutes.

### Option 2: Local Testing

1. **Install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**
   ```bash
   # Edit .env file with your API key
   GOOGLE_API_KEY=your_key_here
   ```

3. **Run**
   ```bash
   streamlit run app.py
   ```

## ✅ Pre-Deployment Checklist

Run the setup check:
```bash
python setup.py
```

This verifies:
- ✅ Python version (3.11+)
- ✅ All dependencies installed
- ✅ Environment variables configured
- ✅ Required directories exist

## 📝 Required Configuration

**Minimum Requirements:**
- `GOOGLE_API_KEY` - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

**Optional:**
- `GROQ_API_KEY` - Alternative LLM
- `NCBI_API_KEY` - Higher PubMed limits

## 🔗 Resources

- [Full README](./README.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)

## 🆘 Troubleshooting

**Setup fails?**
```bash
pip install -r requirements.txt --upgrade
python setup.py
```

**Import errors?**
- Verify Python 3.11+
- Reinstall dependencies

**API errors?**
- Check API key in `.env` or Streamlit secrets
- Verify quota/limits

## 📧 Support

For issues, check:
1. Setup script output
2. Application logs
3. Deployment platform logs

