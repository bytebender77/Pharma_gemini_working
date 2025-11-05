# How to Update Google Gemini API Key in Streamlit Cloud

When your current API key is exhausted or you need to switch to a new key, follow these steps:

## Method 1: Update via Streamlit Cloud Dashboard (Recommended)

### Step 1: Get a New API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key (or use an existing one)
3. Copy the new API key

### Step 2: Update in Streamlit Cloud
1. **Go to Streamlit Cloud Dashboard**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in to your account

2. **Select Your App**
   - Click on your deployed app from the list

3. **Open Secrets Management**
   - Click on **"⚙️ Settings"** (gear icon) in the top right
   - Or go to **"⚙️ Settings" → "Secrets"** from the app menu

4. **Edit Secrets**
   - Click **"Edit secrets"** or the **"✏️ Edit"** button
   - You'll see a TOML editor with your current secrets:
     ```toml
     GOOGLE_API_KEY = "your-old-api-key"
     ```

5. **Update the Key**
   - Replace the old API key with your new one:
     ```toml
     GOOGLE_API_KEY = "your-new-api-key-here"
     ```
   - Click **"Save"** or **"Save secrets"**

6. **Restart the App**
   - After saving, go back to the app settings
   - Click **"⋮"** (three dots menu) → **"Redeploy"**
   - OR click **"Reboot app"** button
   - This restarts the app with the new API key

### Step 3: Verify It Works
1. Go to your app URL
2. Check the sidebar - it should show "✅ Gemini API Connected"
3. Try a simple query to verify:
   - "Get information about aspirin"

---

## Method 2: Update via Streamlit Cloud CLI (Advanced)

If you have Streamlit Cloud CLI installed:

```bash
# Install Streamlit Cloud CLI (if not installed)
pip install streamlit-cloud

# Login to Streamlit Cloud
streamlit cloud login

# Update secrets
streamlit cloud secrets set GOOGLE_API_KEY=your-new-key-here

# Restart app
streamlit cloud restart your-app-name
```

---

## Method 3: Environment Variables (For Self-Hosted)

If you're running your own server (not Streamlit Cloud):

### Docker/Container:
```bash
docker stop your-container
docker run -e GOOGLE_API_KEY=your-new-key-here ... [other options]
```

### Systemd/Service:
```bash
# Edit service file
sudo nano /etc/systemd/system/pharma-ai.service

# Update Environment line:
Environment="GOOGLE_API_KEY=your-new-key-here"

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart pharma-ai
```

### Direct Environment:
```bash
export GOOGLE_API_KEY=your-new-key-here
streamlit run app.py
```

---

## Quick Checklist

- [ ] Get new API key from Google AI Studio
- [ ] Go to Streamlit Cloud dashboard
- [ ] Navigate to app settings → Secrets
- [ ] Update `GOOGLE_API_KEY` value
- [ ] Save changes
- [ ] Restart/Redeploy the app
- [ ] Verify app shows "✅ Gemini API Connected"
- [ ] Test with a sample query

---

## Troubleshooting

### App Still Shows Old Key

**Solution:** The app might be cached. Try:
1. Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. Clear browser cache
3. Wait 30 seconds and refresh (caching delay)

### New Key Not Working

**Check:**
1. ✅ Key is correctly copied (no extra spaces)
2. ✅ Key is valid in Google AI Studio
3. ✅ Key has quota remaining
4. ✅ App was restarted after update

**Debug:**
1. Check app logs in Streamlit Cloud dashboard
2. Look for "Invalid API key" errors
3. Verify key format (should start with "AIza")

### App Won't Restart

**Solution:**
1. Check Streamlit Cloud status page
2. Try redeploying from GitHub (make a small commit)
3. Contact Streamlit support if persistent

---

## Best Practices

### 1. **Monitor API Usage**
- Check Google AI Studio dashboard regularly
- Set up billing alerts in Google Cloud Console
- Monitor quota usage before exhaustion

### 2. **Have Backup Keys Ready**
- Create multiple API keys
- Rotate keys periodically
- Keep unused keys as backups

### 3. **Document Key Rotation**
- Keep track of when keys are rotated
- Note which key is currently active
- Document any issues encountered

### 4. **Test Before Switching**
```bash
# Test new key locally first
export GOOGLE_API_KEY=new-key-here
streamlit run app.py
# Verify it works before deploying
```

### 5. **Use Key Management**
For production, consider:
- Google Secret Manager
- AWS Secrets Manager
- HashiCorp Vault
- Environment variable management tools

---

## Emergency Procedures

### If Key Exhausted During Usage:

1. **Get New Key Immediately**
   - Go to Google AI Studio
   - Create new key
   - Copy key

2. **Quick Update** (2 minutes)
   - Streamlit Cloud → Settings → Secrets
   - Update key
   - Save & Restart

3. **Verify**
   - Check app status
   - Test query

### If Multiple Keys:

You can add multiple keys as backup:

```toml
# In Streamlit secrets
GOOGLE_API_KEY = "primary-key"
GOOGLE_API_KEY_BACKUP = "backup-key-1"
GOOGLE_API_KEY_BACKUP2 = "backup-key-2"
```

Then modify `app.py` to try backups if primary fails (advanced).

---

## Monitoring API Key Status

### In Your App:
- Check sidebar: "✅ Gemini API Connected" = Good
- Check sidebar: "❌ Gemini API Not Connected" = Issue

### In Google AI Studio:
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- See quota usage
- View request history
- Check billing status

---

## Need Help?

If you encounter issues:

1. **Check Streamlit Cloud Logs**
   - Dashboard → App → Logs
   - Look for API key errors

2. **Verify Key Format**
   - Should be ~39 characters
   - Starts with "AIza"

3. **Check Google AI Studio**
   - Verify key is active
   - Check quota/billing

4. **Test Locally**
   - Update `.env` file
   - Run `python setup.py`
   - Run `streamlit run app.py`
   - If works locally, issue is with Streamlit Cloud deployment

---

## Quick Reference

**Streamlit Cloud Secrets URL:**
```
https://share.streamlit.io/ → Your App → ⚙️ Settings → Secrets
```

**Google AI Studio:**
```
https://makersuite.google.com/app/apikey
```

**Key Format:**
```
AIzaSyAbmj5mNoIDzJ5ZvuHsrk61NW3qVkxclkA  (example - 39 characters)
```

