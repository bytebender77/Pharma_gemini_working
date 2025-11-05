# Deployment Guide

This guide covers deploying the Pharma Intelligence AI application to various platforms.

## Prerequisites

- Python 3.11 or higher
- Git (for version control)
- API keys (OpenAI API key is required)

## Deployment Options

### 1. Streamlit Cloud (Recommended - Free)

**Steps:**

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set Main file path: `app.py`
   - Set Python version: `3.11` or higher

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to "Secrets"
   - Add your API keys:
   ```toml
   OPENAI_API_KEY=your_api_key_here
   ```
   - Optionally add:
   ```toml
   NCBI_API_KEY=your_ncbi_key
   ```

4. **Deploy**
   - Click "Deploy!"
   - Your app will be live at `https://your-app-name.streamlit.app`

**Note:** Streamlit Cloud provides free hosting with automatic deployments from GitHub.

---

### 2. Docker Deployment

**Create Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/cache outputs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

**Create .dockerignore:**

```
__pycache__
*.pyc
.env
.git
.gitignore
*.log
data/cache/*
outputs/*
```

**Build and Run:**

```bash
docker build -t pharma-ai .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key_here pharma-ai
```

**Docker Compose:**

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    restart: unless-stopped
```

Run: `docker-compose up -d`

---

### 3. Heroku Deployment

**Create Procfile:**

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**Create runtime.txt:**

```
python-3.11.5
```

**Deploy:**

```bash
heroku create your-app-name
heroku config:set GOOGLE_API_KEY=your_key_here
git push heroku main
```

---

### 4. AWS EC2 / VPS Deployment

**Setup:**

1. **SSH into your server**
2. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3.11 python3-pip git
   ```

3. **Clone and setup:**
   ```bash
   git clone <your-repo-url>
   cd pharma-agentic-ai
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export GOOGLE_API_KEY=your_key_here
   ```

5. **Run with systemd (recommended):**

   Create `/etc/systemd/system/pharma-ai.service`:
   ```ini
   [Unit]
   Description=Pharma Intelligence AI Streamlit App
   After=network.target

   [Service]
   Type=simple
   User=your-user
   WorkingDirectory=/path/to/pharma-agentic-ai
   Environment="GOOGLE_API_KEY=your_key_here"
   ExecStart=/path/to/pharma-agentic-ai/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:
   ```bash
   sudo systemctl enable pharma-ai
   sudo systemctl start pharma-ai
   ```

6. **Configure Nginx (optional):**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

---

### 5. Google Cloud Run

**Create cloudbuild.yaml:**

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/pharma-ai', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/pharma-ai']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'pharma-ai'
      - '--image'
      - 'gcr.io/$PROJECT_ID/pharma-ai'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--set-env-vars'
      - 'GOOGLE_API_KEY=your_key_here'
```

**Deploy:**

```bash
gcloud builds submit --config cloudbuild.yaml
```

---

## Environment Variables for Production

Set these in your deployment platform's environment configuration:

```bash
GOOGLE_API_KEY=required
GROQ_API_KEY=optional
NCBI_API_KEY=optional
DEBUG_MODE=false
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
MAX_SEARCH_RESULTS=20
```

## Post-Deployment Checklist

- [ ] Verify API keys are set correctly
- [ ] Test the application with a simple query
- [ ] Check logs for any errors
- [ ] Verify caching is working
- [ ] Test all agent functions
- [ ] Monitor API rate limits
- [ ] Set up monitoring/alerting (optional)
- [ ] Configure backups for data directory (if persistent)

## Monitoring

### Logs

Check application logs regularly:
- Streamlit Cloud: Dashboard → Logs
- Docker: `docker logs <container-id>`
- Systemd: `journalctl -u pharma-ai -f`

### Performance

Monitor:
- API response times
- Memory usage
- API quota usage
- Error rates

## Troubleshooting Production Issues

1. **High Memory Usage**
   - Reduce `max_iter` in agent configs
   - Enable aggressive caching
   - Increase server resources

2. **API Rate Limits**
   - Check quota usage
   - Implement request queuing
   - Add retry logic with exponential backoff

3. **Slow Response Times**
   - Enable caching
   - Reduce `MAX_SEARCH_RESULTS`
   - Optimize agent configurations

4. **Deployment Failures**
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt
   - Check build logs for errors

## Security Considerations

- Never commit `.env` files
- Use environment variables for secrets
- Enable HTTPS in production
- Implement rate limiting if needed
- Regular dependency updates
- Monitor for security vulnerabilities

