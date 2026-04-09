# Delegation Bundle — G-29: Railway Redis Deployment & Integration

**Agent:** Gemini CLI (DevOps Strike Lead)
**Date:** April 8, 2026
**Priority:** HIGH — Blocks H2H One Win production caching strategy

---

## Mission

Deploy Railway Redis service and integrate it into the MLB Platform for production-grade caching of:
1. OddsAPI responses (smart caching strategy from K-30b)
2. Weather data (OpenWeatherMap forecasts)
3. H2H One Win session state

---

## Railway Deployment Steps

### Step 1: Create Redis Service

**Railway CLI Commands:**
```bash
# Create Redis service (256MB plan)
railway add redis --name "mlb-platform-cache"

# Verify service created
railway services

# Get Redis connection URL
railway variables --service "mlb-platform-cache"
```

**Expected Output:**
- Service ID: `redis-xxxxx`
- REDIS_URL: `redis://default:<password>@host.railway.internal:6379`
- Pricing: $5/month (256MB)

### Step 2: Update Environment Variables

**Add to Railway Project Variables:**
```bash
# Set REDIS_URL in production
railway variable set REDIS_URL "redis://default:<password>@host.railway.internal:6379"

# Verify variable set
railway variables
```

**Also Add Locally:**
```bash
# For local development (use docker-compose or local Redis)
# Add to .env or Railway project variables
REDIS_URL=redis://localhost:6379
```

### Step 3: Verify Redis Connection

**Smoke Test via Railway:**
```bash
# Test Redis connection from production app
railway run python -c "
import os
import redis
from dotenv import load_dotenv

load_dotenv()
redis_url = os.environ.get('REDIS_URL')
r = redis.from_url(redis_url)

# Test connection
r.set('test_key', 'test_value')
result = r.get('test_key')
print(f'Redis connection successful: {result}')
r.delete('test_key')
"
```

**Expected Output:** `Redis connection successful: b'test_value'`

---

## Code Integration Required

### 1. Add Redis Dependency

**File:** `requirements.txt`

**Add:**
```
redis>=5.0.0
```

### 2. Create Redis Client Service

**File:** `backend/services/cache_service.py` (NEW)

```python
"""
Redis cache service for MLB Platform
Handles OddsAPI, weather, and session caching with smart TTL strategies
"""
import os
import json
import redis
from typing import Optional, Any
from datetime import timedelta, datetime
from functools import lru_cache

class CacheService:
    """Redis cache service with TTL-based expiration"""

    def __init__(self):
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.enabled = True
        else:
            self.client = None
            self.enabled = False
            print("Warning: REDIS_URL not set, caching disabled")

    # TTL Strategy (from K-30b)
    TTL_STRATEGY = {
        'far_out': 3600 * 6,      # 6 hours (games >24h away)
        'approaching': 1800,      # 30 min (games 6-24h away)
        'close': 300,             # 5 min (games 2-6h away)
        'live': 60,               # 1 min (games <2h away)
        'weather': 3600 * 3,      # 3 hours (weather forecasts)
        'session': 3600 * 24,     # 24 hours (user sessions)
    }

    def get_odds(self, game_id: str, hours_to_game: float) -> Optional[dict]:
        """Get cached odds data with TTL based on game time"""
        if not self.enabled:
            return None

        cache_key = f"odds:{game_id}"
        cached = self.client.get(cache_key)

        if cached:
            # Check if still fresh based on TTL
            ttl = self._get_odds_ttl(hours_to_game)
            # Redis handles TTL automatically, just return data
            return json.loads(cached)

        return None

    def set_odds(self, game_id: str, odds_data: dict, hours_to_game: float):
        """Cache odds data with appropriate TTL"""
        if not self.enabled:
            return

        cache_key = f"odds:{game_id}"
        ttl = self._get_odds_ttl(hours_to_game)

        self.client.setex(
            cache_key,
            ttl,
            json.dumps(odds_data)
        )

    def get_weather(self, stadium_id: str, game_date: str) -> Optional[dict]:
        """Get cached weather forecast"""
        if not self.enabled:
            return None

        cache_key = f"weather:{stadium_id}:{game_date}"
        cached = self.client.get(cache_key)

        return json.loads(cached) if cached else None

    def set_weather(self, stadium_id: str, game_date: str, weather_data: dict):
        """Cache weather forecast with 3-hour TTL"""
        if not self.enabled:
            return

        cache_key = f"weather:{stadium_id}:{game_date}"
        ttl = self.TTL_STRATEGY['weather']

        self.client.setex(
            cache_key,
            ttl,
            json.dumps(weather_data)
        )

    def get_session(self, user_id: str, session_key: str) -> Optional[dict]:
        """Get user session data"""
        if not self.enabled:
            return None

        cache_key = f"session:{user_id}:{session_key}"
        cached = self.client.get(cache_key)

        return json.loads(cached) if cached else None

    def set_session(self, user_id: str, session_key: str, session_data: dict):
        """Cache user session with 24-hour TTL"""
        if not self.enabled:
            return

        cache_key = f"session:{user_id}:{session_key}"
        ttl = self.TTL_STRATEGY['session']

        self.client.setex(
            cache_key,
            ttl,
            json.dumps(session_data)
        )

    def delete_session(self, user_id: str, session_key: str):
        """Delete user session (logout)"""
        if not self.enabled:
            return

        cache_key = f"session:{user_id}:{session_key}"
        self.client.delete(cache_key)

    def _get_odds_ttl(self, hours_to_game: float) -> int:
        """Determine TTL based on time to game"""
        if hours_to_game > 24:
            return self.TTL_STRATEGY['far_out']
        elif hours_to_game > 6:
            return self.TTL_STRATEGY['approaching']
        elif hours_to_game > 2:
            return self.TTL_STRATEGY['close']
        else:
            return self.TTL_STRATEGY['live']

    def clear_all_odds_cache(self):
        """Clear all odds cache (use sparingly)"""
        if not self.enabled:
            return

        # Scan and delete all keys matching "odds:*"
        for key in self.client.scan_iter("odds:*"):
            self.client.delete(key)

    def get_cache_stats(self) -> dict:
        """Return cache statistics for monitoring"""
        if not self.enabled:
            return {"enabled": False}

        info = self.client.info('stats')
        return {
            "enabled": True,
            "total_keys": self.client.dbsize(),
            "hits": info.get('keyspace_hits', 0),
            "misses": info.get('keyspace_misses', 0),
            "hit_rate": info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
        }

# Singleton instance
_cache_service = None

def get_cache_service() -> CacheService:
    """Get or create CacheService singleton"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
```

### 3. Update OddsAPI Client (Use Cache)

**File:** `backend/services/odds.py` (MODIFY)

```python
# Add import
from backend.services.cache_service import get_cache_service

# In get_odds() function, add caching layer:
def get_odds(game_id: str, hours_to_game: float) -> dict:
    """Get odds with Redis caching"""
    cache = get_cache_service()

    # Check cache first
    cached_odds = cache.get_odds(game_id, hours_to_game)
    if cached_odds:
        return cached_odds

    # Cache miss - call OddsAPI
    odds_data = odds_api_client.get_odds(game_id)

    # Store in cache
    cache.set_odds(game_id, odds_data, hours_to_game)

    return odds_data
```

### 4. Update Weather Client (Use Cache)

**File:** `backend/services/park_weather.py` (MODIFY)

```python
# Add import
from backend.services.cache_service import get_cache_service

# In get_weather_forecast() function, add caching:
def get_weather_forecast(stadium_id: str, game_date: str) -> dict:
    """Get weather with Redis caching"""
    cache = get_cache_service()

    # Check cache first
    cached_weather = cache.get_weather(stadium_id, game_date)
    if cached_weather:
        return cached_weather

    # Cache miss - call OpenWeatherMap API
    weather_data = openweathermap_client.get_forecast(stadium_id, game_date)

    # Store in cache
    cache.set_weather(stadium_id, game_date, weather_data)

    return weather_data
```

---

## Testing Required

### Unit Tests

**File:** `tests/test_cache_service.py` (NEW)

```python
import pytest
from backend.services.cache_service import CacheService

def test_cache_service_initialization():
    """Test CacheService can initialize"""
    cache = CacheService()
    assert cache is not None

def test_odds_caching_ttl_strategy():
    """Test TTL strategy based on game time"""
    cache = CacheService()

    # Far out game (>24h)
    ttl_far = cache._get_odds_ttl(30)
    assert ttl_far == 3600 * 6  # 6 hours

    # Approaching game (6-24h)
    ttl_approaching = cache._get_odds_ttl(12)
    assert ttl_approaching == 1800  # 30 min

    # Close game (2-6h)
    ttl_close = cache._get_odds_ttl(4)
    assert ttl_close == 300  # 5 min

    # Live game (<2h)
    ttl_live = cache._get_odds_ttl(1)
    assert ttl_live == 60  # 1 min

def test_weather_caching():
    """Test weather forecast caching"""
    cache = CacheService()

    # Set weather
    cache.set_weather("COL", "2026-04-10", {
        "temp_f": 75.0,
        "wind_speed_mph": 10.0
    })

    # Get weather
    weather = cache.get_weather("COL", "2026-04-10")
    assert weather["temp_f"] == 75.0
    assert weather["wind_speed_mph"] == 10.0
```

### Integration Tests (Railway)

```bash
# Test Redis connection on Railway
railway run python -c "
from backend.services.cache_service import get_cache_service

cache = get_cache_service()

# Test basic set/get
cache.set_odds('game_123', {'total': 9.5}, hours_to_game=4)
odds = cache.get_odds('game_123', 4)
print(f'Odds cache test: {odds}')

# Test cache stats
stats = cache.get_cache_stats()
print(f'Cache stats: {stats}')
"
```

---

## Verification Checklist

- [ ] Redis service created on Railway ($5/month plan)
- [ ] REDIS_URL environment variable set in Railway project
- [ ] `redis>=5.0.0` added to requirements.txt
- [ ] `backend/services/cache_service.py` created
- [ ] OddsAPI client updated to use cache
- [ ] Weather client updated to use cache
- [ ] Unit tests pass: `pytest tests/test_cache_service.py -q`
- [ ] Railway smoke test passes
- [ ] Cache hit rate >50% after 1 week (monitoring)

---

## Estimated Work

**Gemini CLI Tasks:**
1. Railway Redis deployment: 30 minutes
2. Environment variable configuration: 15 minutes
3. Redis connection verification: 15 minutes
4. Code review (cache_service.py): 30 minutes
5. Integration testing: 30 minutes
6. HANDOFF.md update: 15 minutes

**Total Time:** ~2-3 hours

**Claude Code Tasks:**
- Write cache_service.py implementation (or delegate to Kimi?)
- Update odds.py, park_weather.py to use cache
- Write unit tests
- Handoff to Gemini for deployment

---

## Deliverables

1. **Railway Redis service deployed** (256MB, $5/month)
2. **REDIS_URL configured** in production and local
3. **cache_service.py** created and tested
4. **Cache integration** in odds.py, park_weather.py
5. **Monitoring dashboard** (cache hit rate in admin panel)
6. **HANDOFF.md updated** with Redis status

---

## Escalation

If you encounter:
- **Redis connection fails** → Check REDIS_URL format, Railway service status
- **Cache not persisting** → Check Redis memory limits (256MB should be sufficient)
- **High cache miss rate** → Review TTL strategy, adjust if needed
- **Railway pricing issue** → User approval required for $5/month

---

**Assigned to:** Gemini CLI (DevOps Strike Lead)
**Review required:** Claude Code (code integration)
**Sign-off:** User (approve $5/month Redis cost)

*Last Updated: April 8, 2026 — Session S29*
