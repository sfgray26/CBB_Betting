---
name: waiver_recovery
agent: kimi-cli
description: |
  Resilient Yahoo Fantasy API error recovery with fallback strategies,
  circuit breaker pattern, and graceful degradation for waiver/lineup operations.
  
  Designed for CBB Edge Analyzer's fantasy baseball integration.
tools: ["Read", "Edit", "Write", "Grep", "Bash"]
---

# Waiver Recovery Skill

## Purpose

Handle Yahoo Fantasy API failures gracefully. When `percent_owned` subresource 
fails or lineup positions mismatch, this skill provides structured recovery 
strategies instead of 503 errors and skipped players.

## Error Patterns We Handle

### Pattern 1: Invalid Subresource (percent_owned)
```
Yahoo API error 400: Invalid subresource percent_owned requested
```

### Pattern 2: Position Validation Failures
```
Skipped Marcus Semien (pos=2B): game_id mismatch
```

### Pattern 3: Token Refresh Cascades
```
Yahoo tokens refreshed and persisted to .env  # but still fails
```

## Recovery Strategies

### Strategy A: Metadata-Only Fallback (Immediate)
When `percent_owned` fails, fetch `metadata` only and estimate ownership:

```python
async def fetch_waiver_players_fallback(league_id: str) -> List[Player]:
    """
    Fallback when percent_owned subresource is unavailable.
    Uses ADP data as ownership proxy.
    """
    try:
        # Primary: Try with percent_owned
        return await yahoo.get_players(
            league_id, 
            subresources="metadata,percent_owned"
        )
    except YahooAPIError as e:
        if "percent_owned" in str(e):
            logger.warning("percent_owned unavailable, using ADP proxy")
            
            # Fallback: Metadata only + ADP lookup
            players = await yahoo.get_players(
                league_id,
                subresources="metadata"  # Always available
            )
            
            # Enrich with local ADP data
            for player in players:
                player.estimated_percent_owned = estimate_ownership_from_adp(
                    player.name, 
                    adp_data=load_adp_data()
                )
            
            return players
        raise
```

### Strategy B: Circuit Breaker (Reliability)
After N consecutive Yahoo API failures, switch to cached/offline mode:

```python
class YahooCircuitBreaker:
    """
    Circuit breaker for Yahoo Fantasy API.
    State transitions: CLOSED -> OPEN -> HALF_OPEN
    """
    
    FAILURE_THRESHOLD = 3
    RECOVERY_TIMEOUT = 300  # 5 minutes
    
    def __init__(self):
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Yahoo API circuit open, using cached data")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()
        if self.failures >= self.FAILURE_THRESHOLD:
            self.state = "OPEN"
            logger.error(f"Yahoo API circuit OPEN after {self.failures} failures")
    
    def _on_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def _should_attempt_reset(self) -> bool:
        return (datetime.now() - self.last_failure_time).seconds > self.RECOVERY_TIMEOUT
```

### Strategy C: Position Normalization (Lineup Fixes)
Handle position mismatches between Steamer projections and Yahoo's roster:

```python
class PositionNormalizer:
    """
    Normalize position eligibility between projection sources and Yahoo.
    """
    
    # Map Steamer positions to Yahoo's expected format
    POSITION_MAP = {
        "C": "C",
        "1B": "1B", 
        "2B": "2B",
        "3B": "3B",
        "SS": "SS",
        "LF": "LF",
        "CF": "CF",
        "RF": "RF",
        "OF": "LF",  # Generic OF maps to LF as fallback
        "DH": "Util",  # Designated hitter -> Utility slot
        "SP": "SP",
        "RP": "RP",
    }
    
    @classmethod
    def normalize_lineup(cls, optimized_lineup: Dict, yahoo_roster: Dict) -> Dict:
        """
        Match optimized lineup to Yahoo's actual roster slots.
        Returns: {yahoo_slot_id: player_id} with validated positions
        """
        assignments = {}
        used_players = set()
        
        for slot in yahoo_roster["slots"]:
            slot_pos = slot["position"]
            
            # Find best matching player from optimization
            for opt_player in optimized_lineup["starters"]:
                if opt_player["id"] in used_players:
                    continue
                    
                # Check position eligibility (Yahoo's view wins)
                if slot_pos in opt_player.get("yahoo_positions", []):
                    assignments[slot["id"]] = opt_player["id"]
                    used_players.add(opt_player["id"])
                    break
            else:
                # No valid player found — log for review
                logger.warning(f"No eligible player for slot {slot_pos}")
        
        return assignments
    
    @classmethod
    def validate_before_submit(cls, assignments: Dict, yahoo_roster: Dict) -> bool:
        """
        Dry-run validation before calling Yahoo's set_lineup API.
        Returns True if all assignments are valid.
        """
        errors = []
        
        for slot_id, player_id in assignments.items():
            slot = next(s for s in yahoo_roster["slots"] if s["id"] == slot_id)
            player = next(p for p in yahoo_roster["players"] if p["id"] == player_id)
            
            if slot["position"] not in player.get("eligible_positions", []):
                errors.append(
                    f"Invalid: {player['name']} ({player['positions']}) in {slot['position']}"
                )
        
        if errors:
            logger.error("Lineup validation failed:\n" + "\n".join(errors))
            return False
        return True
```

### Strategy D: Stale Cache Fallback (Availability)
When all else fails, serve last-known-good data:

```python
class StaleCacheManager:
    """
    Manages fallback to cached data when APIs fail.
    """
    
    CACHE_TTL = timedelta(hours=24)  # Accept data up to 24h old
    
    def __init__(self, cache_dir: str = ".cache/fantasy"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def get_with_fallback(self, key: str, fetch_func: Callable) -> Any:
        """
        Try live fetch first, fallback to cache on failure.
        """
        try:
            data = await fetch_func()
            self._write_cache(key, data)
            return {"data": data, "fresh": True, "source": "api"}
        except Exception as e:
            logger.warning(f"API failed ({e}), checking cache for {key}")
            
            cached = self._read_cache(key)
            if cached and self._is_acceptable(cached["timestamp"]):
                return {
                    "data": cached["data"], 
                    "fresh": False, 
                    "source": "cache",
                    "age_hours": self._age_hours(cached["timestamp"])
                }
            
            raise NoDataAvailableError(f"API failed and no acceptable cache for {key}")
    
    def _write_cache(self, key: str, data: Any):
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f)
    
    def _read_cache(self, key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        with open(cache_file) as f:
            return json.load(f)
    
    def _is_acceptable(self, timestamp: str) -> bool:
        ts = datetime.fromisoformat(timestamp)
        return datetime.now() - ts < self.CACHE_TTL
```

## Integration: Enhanced Yahoo Client

```python
# backend/fantasy_baseball/yahoo_client_resilient.py

class ResilientYahooClient(YahooClient):
    """
    Yahoo client with circuit breaker, fallback strategies, and stale cache.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit = YahooCircuitBreaker()
        self.cache = StaleCacheManager()
        self.positions = PositionNormalizer()
    
    async def get_waiver_players(self, league_id: str, **filters) -> WaiverResponse:
        """
        Get waiver wire players with full fallback chain.
        """
        cache_key = f"waiver_{league_id}_{hash(str(filters))}"
        
        try:
            # Attempt 1: Circuit breaker wrapped API call
            result = self.circuit.call(
                self._fetch_waiver_with_fallback,
                league_id,
                filters
            )
            return WaiverResponse(
                players=result,
                source="yahoo_api",
                fresh=True,
                errors=[]
            )
            
        except CircuitOpenError:
            # Attempt 2: Circuit open, use cache
            cached = self.cache._read_cache(cache_key)
            if cached:
                return WaiverResponse(
                    players=cached["data"],
                    source="cache",
                    fresh=False,
                    errors=["Circuit open: serving stale data"]
                )
            raise
    
    async def _fetch_waiver_with_fallback(self, league_id, filters):
        """Internal: Try primary, fallback to metadata-only."""
        try:
            return await super().get_waiver_players(league_id, **filters)
        except YahooAPIError as e:
            if "percent_owned" in str(e).lower():
                # Fallback: metadata only + ADP enrichment
                return await self._fetch_waiver_metadata_only(league_id, filters)
            raise
    
    async def set_lineup_resilient(self, team_id: str, lineup: Dict) -> LineupResult:
        """
        Set lineup with pre-validation and graceful degradation.
        """
        # Step 1: Get current Yahoo roster
        yahoo_roster = await self.get_roster(team_id)
        
        # Step 2: Normalize positions
        normalized = self.positions.normalize_lineup(lineup, yahoo_roster)
        
        # Step 3: Validate before hitting API
        if not self.positions.validate_before_submit(normalized, yahoo_roster):
            return LineupResult(
                success=False,
                errors=["Position validation failed"],
                retry_possible=False
            )
        
        # Step 4: Attempt with circuit breaker
        try:
            result = self.circuit.call(
                super().set_lineup,
                team_id,
                normalized
            )
            return LineupResult(success=True, changes=result)
        except Exception as e:
            logger.error(f"Lineup set failed: {e}")
            return LineupResult(
                success=False,
                errors=[str(e)],
                retry_possible=True,
                suggested_action="Retry in 5 minutes or set manually"
            )
```

## API Response Schema

All recovery strategies return standardized responses:

```python
class WaiverResponse(BaseModel):
    players: List[Player]
    source: Literal["yahoo_api", "cache", "projection_estimate"]
    fresh: bool
    errors: List[str]
    metadata: Dict = {}

class LineupResult(BaseModel):
    success: bool
    changes: Optional[Dict] = None
    errors: List[str] = []
    retry_possible: bool = False
    suggested_action: Optional[str] = None
```

## Monitoring & Alerts

```python
# Log structured events for monitoring
logger.info("waiver_recovery_strategy_used", extra={
    "strategy": "metadata_fallback",
    "league_id": league_id,
    "players_returned": len(players),
    "estimated_data": True
})
```

## Testing

```python
@pytest.mark.asyncio
async def test_percent_owned_fallback():
    """Verify graceful degradation when percent_owned is unavailable."""
    client = ResilientYahooClient()
    
    # Mock API to fail on percent_owned
    with mock.patch('yahoo_api.call', side_effect=YahooAPIError("percent_owned")):
        result = await client.get_waiver_players("mlb.l.12345")
    
    assert result.source == "yahoo_api"  # Still from API
    assert result.fresh is True
    assert all(p.estimated_percent_owned is not None for p in result.players)

@pytest.mark.asyncio  
async def test_circuit_opens_after_failures():
    """Circuit breaker opens after threshold failures."""
    client = ResilientYahooClient()
    client.circuit.FAILURE_THRESHOLD = 2
    
    with mock.patch('yahoo_api.call', side_effect=Exception("Network error")):
        await client.get_waiver_players("mlb.l.12345")  # Fail 1
        await client.get_waiver_players("mlb.l.12345")  # Fail 2
        
        with pytest.raises(CircuitOpenError):
            await client.get_waiver_players("mlb.l.12345")  # Circuit open
```

## Migration Path

1. **Phase 1** (Immediate): Add `ResilientYahooClient` alongside existing client
2. **Phase 2** (Testing): A/B test with 10% of requests
3. **Phase 3** (Full): Replace default client, keep old as fallback

---

*Created from analysis of production logs showing Yahoo API instability.*
*References: logs/2026-03-26, backend/fantasy_baseball/yahoo_client.py*
