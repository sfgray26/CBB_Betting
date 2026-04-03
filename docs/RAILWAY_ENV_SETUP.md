# Railway Environment Setup

## Setting the OpenWeather API Key

### 1. Get Your API Key

1. Go to https://openweathermap.org/api
2. Create a free account
3. Navigate to "My API Keys"
4. Copy your default key (or generate a new one)

### 2. Add to Railway

**Option A: Railway Dashboard (Recommended)**

1. Go to https://railway.app/dashboard
2. Select your project
3. Click on your service (the API/backend)
4. Go to the **Variables** tab
5. Click **+ New Variable**
6. Add:
   - Name: `OPENWEATHER_API_KEY`
   - Value: `your_api_key_here`
7. Click **Add**
8. Railway will auto-redeploy

**Option B: Railway CLI**

```bash
# Install Railway CLI if not already installed
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Set the variable
railway variables set OPENWEATHER_API_KEY=your_api_key_here

# Verify
railway variables
```

### 3. Verify It's Working

After deployment, check the logs:

```bash
railway logs
```

You should see weather data being fetched:
```
INFO:weather_fetcher:Fetched weather for Wrigley Field: 72°F, 12mph WSW
INFO:park_weather:Wrigley Field: 💨 Wind helps (12mph WSW) | ☀️ Hot (78°F)
```

### 4. Test the Endpoint

```bash
curl "https://your-api.railway.app/api/fantasy/lineup/2025-03-27" \
  -H "X-API-Key: your_api_key"
```

Check that pitchers now show proper opponent implied runs and park factors.

## Other Required Variables

Make sure these are also set in Railway:

| Variable | Source | Purpose |
|----------|--------|---------|
| `DATABASE_URL` | Railway Postgres | Database connection |
| `THE_ODDS_API_KEY` | the-odds-api.com | Sports betting odds |
| `YAHOO_CLIENT_ID` | Yahoo Developer | Fantasy baseball auth |
| `YAHOO_CLIENT_SECRET` | Yahoo Developer | Fantasy baseball auth |
| `YAHOO_REFRESH_TOKEN` | Auth flow | Fantasy baseball auth |
| `OPENWEATHER_API_KEY` | OpenWeatherMap | Weather data |

## Deployment Isolation Flags

Use these when separating the current CBB production deployment from a new MLB UAT Railway project.

| Variable | CBB Production | MLB UAT | Purpose |
|----------|----------------|---------|---------|
| `DEPLOYMENT_ROLE` | `cbb-prod` | `mlb-uat` | Labels the deployment in `/` and `/health` responses |
| `ENABLE_MAIN_SCHEDULER` | `true` | `false` for first safe boot | Controls the main APScheduler loop in `backend/main.py` |
| `ENABLE_STARTUP_CATCHUP` | `true` | `false` | Prevents post-3AM CBB catch-up analysis from running in UAT |
| `ENABLE_INGESTION_ORCHESTRATOR` | current intended value | `false` until verified | Controls separate MLB ingestion scheduler |
| `ENABLE_MLB_ANALYSIS` | `false` during CBB freeze | `false` until verified | Prevents MLB scheduled analysis before UAT is ready |
| `RUN_STARTUP_MIGRATIONS` | `true` | `true` only against the UAT database | Controls boot-time migration scripts in the Docker container |
| `RUN_STARTUP_DB_INIT` | `true` | `true` only against the UAT database | Controls `python -m backend.models` schema init on boot |

### Recommended First-Boot Posture For MLB UAT

Set these before the first UAT deploy:

```bash
DEPLOYMENT_ROLE=mlb-uat
ENABLE_MAIN_SCHEDULER=false
ENABLE_STARTUP_CATCHUP=false
ENABLE_INGESTION_ORCHESTRATOR=false
ENABLE_MLB_ANALYSIS=false
RUN_STARTUP_MIGRATIONS=true
RUN_STARTUP_DB_INIT=true
```

That gives you a safe application boot against an isolated UAT database without starting the shared production scheduler or triggering CBB startup catch-up behavior.

## GitHub + Railway Cutover Sequence

1. Tag the current production-safe CBB commit.

```bash
git tag -a cbb-prod-v1.0 -m "Frozen CBB production baseline before MLB UAT split"
git push origin cbb-prod-v1.0
```

2. Create a protected production branch for CBB.

```bash
git checkout -b stable/cbb-prod
git push -u origin stable/cbb-prod
```

3. In Railway, repoint the existing live CBB project to `stable/cbb-prod`.

4. Create a separate Railway project for MLB UAT with its own Postgres service and its own variables.

5. Point MLB UAT at `main` or a dedicated branch such as `mlb/uat`, never at `stable/cbb-prod`.

6. Verify the MLB UAT `DATABASE_URL` is different from CBB production before the first deploy.

## Troubleshooting

### "Weather unavailable" in logs
- Check `OPENWEATHER_API_KEY` is set correctly
- Verify key is active at https://home.openweathermap.org/api_keys
- Free tier has 1,000 calls/day limit

### Still using estimates
- Check logs for `OPENWEATHER_API_KEY not set, using estimates`
- Variable may not be deployed yet - wait for green deployment

### API limit exceeded
- Free tier: 1,000 calls/day
- If exceeded, system falls back to seasonal estimates
- Consider upgrading to Base plan ($0.15/100 calls) during season
