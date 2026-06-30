# OAuth Configuration & Authentication

## Two-Layer Authentication

The platform uses two layers of authentication:

### Layer 1: API Key (Application Layer)
- **Purpose**: Protect the application from unauthorized access
- **Implementation**: `X-API-Key` header validated on each request
- **Credential**: `API_KEY_USER1` (Railway environment variable)
- **Login Flow**: Visit `/login` and enter your API key

### Layer 2: Yahoo OAuth (Data Layer)
- **Purpose**: Access Yahoo Fantasy Sports data
- **Implementation**: OAuth 2.0 with automatic token refresh
- **Credentials**: `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, `YAHOO_REFRESH_TOKEN`, `YAHOO_LEAGUE_ID`
- **Status**: ✅ Configured and working on Railway

---

## Current Configuration (Railway)

### Yahoo OAuth Environment Variables
```
YAHOO_CLIENT_ID=dj0yJmk9M09lWjdMczhqeXR2JmQ9WVdr...
YAHOO_CLIENT_SECRET=13d2747c85e38363ffcd68ec6d4c8d51
YAHOO_REFRESH_TOKEN=ABvAsGm6xEQPHJbjci6kYtQGSCaZ~001...
YAHOO_LEAGUE_ID=72586
YAHOO_ACCESS_TOKEN=iyL8W36fvQXTCaWaePsErnRXKWJc2S.r... (auto-refreshed)
```

### API Key Environment Variable
```
API_KEY_USER1=j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg
```

---

## How to Use the Platform

### Step 1: Get Your API Key
Contact the platform administrator for your API key, or check the Railway environment variables if you have admin access.

### Step 2: Log In
1. Visit `https://fantasy-app-production-5079.up.railway.app/login`
2. Enter your API key
3. The system validates your key and sets a session cookie

### Step 3: Access Protected Modules
Once logged in, all modules work:
- **Dashboard**: Scoreboard, streaks, waiver targets
- **War Room**: Matchup analysis, verdicts, recommendations
- **My Roster**: Roster management, "This Week" stats
- **Waiver Wire**: Player recommendations, ownership%
- **Budget**: Constraint tracking, IP pace, season adds
- **Streaming**: Two-start pitcher recommendations

---

## Protected Endpoints (Require API Key)

| Endpoint | Purpose | Data Source |
|-----------|---------|-------------|
| `/api/fantasy/roster` | My Roster | Yahoo OAuth ✅ |
| `/api/fantasy/budget` | Budget Panel | Yahoo OAuth ✅ |
| `/api/fantasy/matchup` | War Room | Yahoo OAuth ✅ |
| `/api/fantasy/waiver` | Waiver Wire | Yahoo OAuth ✅ |
| `/api/fantasy/scoreboard` | Dashboard | Yahoo OAuth ✅ |

## Public Endpoints (No API Key Required)

| Endpoint | Purpose |
|-----------|---------|
| `/api/fantasy/global-freshness` | System freshness status |
| `/api/fantasy/streaming/recommendations` | Streaming recommendations |
| `/api/fantasy/matchup-preview` | Opponent preview |
| `/health` | Health check |

---

## Troubleshooting

### "API key required" (401)
**Cause**: No API key provided or invalid API key  
**Solution**: Log in at `/login` with a valid API key

### "Invalid API key" (401)
**Cause**: API key doesn't match `API_KEY_USER1` environment variable  
**Solution**: Contact administrator for correct API key

### "Yahoo client not initialized"
**Cause**: Yahoo OAuth credentials missing or invalid  
**Solution**: Verify Railway environment variables are set correctly

### "Token refresh failed"
**Cause**: Yahoo refresh token expired  
**Solution**: Regenerate refresh token via Yahoo OAuth flow (requires user interaction)

---

## Token Regeneration (If Refresh Token Expires)

If the Yahoo refresh token expires, follow these steps:

1. **Local Setup**:
   ```bash
   git clone <repo>
   cd <repo>
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run OAuth Flow**:
   ```bash
   python scripts/auth/yahoo_oauth.py
   ```

3. **Complete Yahoo Authorization**:
   - Copy the authorization URL
   - Open in browser
   - Grant permissions
   - Copy the authorization code
   - Paste back into the script

4. **Update Railway Variables**:
   - Copy the new `YAHOO_REFRESH_TOKEN` from the script output
   - Update Railway environment variable via dashboard or CLI:
     ```bash
     railway variables set YAHOO_REFRESH_TOKEN <new_token>
     ```

---

## Status

- ✅ Yahoo OAuth configured and working
- ✅ Automatic token refresh enabled
- ✅ API key authentication functional
- ✅ All protected endpoints returning data when authenticated

**Last Verified**: 2026-06-26
