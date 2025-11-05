# ⚡ Quick API Key Update - Streamlit Cloud

**Time: ~2 minutes**

## Steps

1. **Get New Key**
   - Visit: https://makersuite.google.com/app/apikey
   - Copy your new API key

2. **Update in Streamlit Cloud**
   - Go to: https://share.streamlit.io
   - Click your app
   - Click **⚙️ Settings** → **Secrets**
   - Click **Edit secrets**
   - Change: `GOOGLE_API_KEY = "your-new-key"`
   - Click **Save**

3. **Restart App**
   - Go back to app settings
   - Click **⋮** → **Redeploy**
   - OR click **Reboot app**

4. **Verify**
   - Check sidebar shows: ✅ Gemini API Connected
   - Test with: "Get information about aspirin"

## ✅ Done!

Your app is now using the new API key.

---

**Need more details?** See [UPDATE_API_KEY.md](./UPDATE_API_KEY.md)

