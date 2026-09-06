# Deployment Guide — Streamlit Cloud

## Quick Start (5 minutes)

### Step 1: Ensure Code is on GitHub
✅ Already done! Your code is at:
```
https://github.com/madhavarao-avineni/ai-research-agent
```

### Step 2: Deploy to Streamlit Cloud

1. **Visit Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub (or create account)

2. **Create New App**
   - Click "New app"
   - Select repository: `madhavarao-avineni/ai-research-agent`
   - Select branch: `master`
   - Select main file: `ui/app.py`

3. **Configure Secrets**
   - After deployment, go to **Settings → Secrets**
   - Copy the contents of `.streamlit/secrets.toml.example`
   - Paste into the secrets editor
   - Fill in your actual API keys:
     ```toml
     OPENROUTER_API_KEY = "sk-or-v1-your-actual-key"
     TAVILY_API_KEY = "tvly-dev-your-actual-key"
     NEWSAPI_KEY = "your-actual-key"
     LLM_MODEL = "openai/gpt-4o-mini"
     ```

4. **Reboot App**
   - Click **Settings → Reboot app**
   - App will restart with secrets loaded

5. **Done! 🎉**
   - Your app is live at: `https://<username>-ai-research-agent-<hash>.streamlit.app`

---

## Configuration Files

### `.streamlit/config.toml`
Defines the app theme and settings:
- **Colors**: Brutalist SaaS design (charcoal + golden yellow)
- **Server**: Headless, CORS disabled, secure
- **UI**: Minimal toolbar, no footer icons

### `.streamlit/secrets.toml.example`
Template for API keys. **Do NOT commit actual keys.**

---

## Environment Variables

Required for full functionality:

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `OPENROUTER_API_KEY` | ✅ | `sk-or-v1-...` | LLM provider (OpenRouter) |
| `LLM_MODEL` | ✅ | `openai/gpt-4o-mini` | Model selection |
| `TAVILY_API_KEY` | ✅ | `tvly-dev-...` | Web search |
| `NEWSAPI_KEY` | ⚠️ | `your-key` | News retrieval (optional) |
| `LANGCHAIN_TRACING_V2` | ❌ | `false` | Observability (optional) |

---

## Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] GitHub repository is public
- [ ] Streamlit Cloud account created
- [ ] New app created (repo, branch, file selected)
- [ ] API keys added to Secrets
- [ ] App rebooted
- [ ] Test app loads successfully
- [ ] Test research query works
- [ ] Share app URL with team

---

## Usage

Once deployed:

1. **Access the App**
   - Open: `https://<your-app>.streamlit.app`
   - Share link with team

2. **Run Research**
   - Enter research question
   - Adjust max iterations in sidebar
   - Click "Begin Research"
   - Wait for results (typically 30-60 seconds)

3. **View Results**
   - **Report**: Full markdown research report
   - **Sources**: All retrieved sources with rankings
   - **Contradictions**: Conflicting evidence
   - **Insights**: Evidence-backed findings

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
**Solution**: `requirements.txt` is missing dependencies.
- Check `.streamlit/logs/` for details
- Add missing package to `requirements.txt`
- Commit and push
- Streamlit will auto-rebuild

### "API key invalid" Error
**Solution**: Secrets not properly configured.
- Go to **Settings → Secrets**
- Verify key format (no extra quotes/spaces)
- Check key is active on provider's dashboard
- Reboot app

### "Research query times out"
**Solution**: Model might be overloaded.
- Try a simpler question
- Reduce `MAX_RESEARCH_ITERATIONS` in secrets
- Try a different LLM model
- Check API provider's status page

### "No sources retrieved"
**Solution**: Search APIs might be failing.
- Verify `TAVILY_API_KEY` and `NEWSAPI_KEY` in secrets
- Check API quotas on provider dashboards
- App will fall back to LLM-only research (graceful degradation)

---

## Performance

**Typical Execution Timeline**:
- Query submission: <1s
- Planner (question decomposition): 2-5s
- Retriever (parallel search): 2-10s
- Analyzer (critical examination): 1-3s
- Insight generator: 3-8s
- Fact checker: 1-2s
- Report builder: 5-10s
- **Total**: 15-40s per research query

**Scaling**:
- Streamlit Cloud automatically scales
- Free tier: 1 app, 1 concurrent user
- Upgraded: Multiple apps, higher concurrency

---

## Updating the App

1. **Make changes locally**
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push origin master
   ```

2. **Streamlit Cloud auto-deploys**
   - Detects push to GitHub
   - Rebuilds app automatically
   - Live in ~1-2 minutes

3. **Check deployment**
   - View logs in Streamlit Cloud dashboard
   - Test the app

---

## Support & Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Cloud Docs**: https://docs.streamlit.io/deploy/streamlit-cloud
- **GitHub Repo**: https://github.com/madhavarao-avineni/ai-research-agent
- **Project Status**: Production-ready, fully tested

---

**Deployment Date**: 2026-09-06  
**Status**: Ready for Streamlit Cloud  
**Testing**: 19/19 unit tests passing ✅
