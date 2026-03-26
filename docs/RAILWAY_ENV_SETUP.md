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
