# K-B Research: Player Identity Resolution Design

**Date:** 2026-04-01 (Corrected 2026-04-01)  
**Researcher:** Kimi CLI  
**Status:** CORRECTED — Critical inaccuracies fixed based on verified fixtures  
**Research embargo:** NO production code until specs approved by Claude.

---

## ⚠️ CORRECTION NOTICE (2026-04-01)

**Previous versions of this report contained a critical error:**
- ❌ **WRONG:** Claimed Yahoo `player_key` format was `"mlb.p.{mlbam_id}"` and `player_id` was mlbam ID
- ✅ **CORRECT:** Yahoo `player_key` format is `{game_id}.p.{yahoo_player_id}` (e.g., `"469.p.7590"`) and `player_id` is a **proprietary Yahoo ID** that does NOT match mlbam ID

**Evidence:** `tests/fixtures/yahoo_adp_injury.json` shows:
```json
{
  "player_key": "469.p.7590",
  "player_id": "7590",
  "name": "Justin Verlander",
  ...
}
```

**Impact:** Identity resolution is MORE complex than initially assessed — requires name-based mapping rather than direct ID extraction.

---

## 1. Executive Summary

Three distinct player ID systems are in use across the Fantasy Baseball platform:

| System | Format | Source | Usage |
|--------|--------|--------|-------|
| **BDL player.id** | `int` | BallDontLie API | Games, injuries, teams endpoints |
| **mlbam ID** | `int` | MLB Stats API / Baseball Savant | Statcast, industry standard |
| **Yahoo player_id** | `str` | Yahoo Fantasy API | **Proprietary** Yahoo-only ID |

**Critical Finding:** Yahoo uses **proprietary player IDs** that do NOT match mlbam IDs or any external system ([source](YAHOO_FANTASY_API_RESEARCH.md): "Yahoo uses proprietary player IDs that don't match other sources").

**Resolution Path:** Name-based lookup via `pybaseball.playerid_lookup()` or cached mapping table.

---

## 2. ID System Deep Dive

### 2.1 BDL player.id (Internal)

```python
# From backend/data_contracts/mlb_player.py
class MLBPlayer(BaseModel):
    id: int              # BDL internal ID (e.g., 208 for Shohei Ohtani)
    first_name: str
    last_name: str
    full_name: str      # "Shohei Ohtani"
    # ... NO mlbam_id field present ...
```

**Verified fixture:** `tests/fixtures/bdl_mlb_players.json`
```json
{
  "id": 208,
  "first_name": "Shohei",
  "last_name": "Ohtani",
  "full_name": "Shohei Ohtani",
  ...
}
```

**Characteristics:**
- Integer, internal to BDL
- Used in: `MLBGame`, `MLBInjury`, all BDL endpoints
- **Not directly compatible** with Yahoo or MLBAM systems

### 2.2 mlbam ID (Industry Standard)

```python
# From backend/fantasy_baseball/statcast_scraper.py
def search_player_id(name: str) -> Optional[int]:
    """Search for MLBAM player ID by name."""
    params = {"search": name}
    data = _get_cached_or_fetch("player/search", params, use_cache=False)
    if data.get("players"):
        return data["players"][0].get("mlbam_id")  # Official MLB ID
    return None
```

**Characteristics:**
- Integer, MLB Advanced Media identifier
- Used by: MLB Stats API, Baseball Savant, pybaseball
- **Industry standard** for cross-platform player identification
- **NOT the same as Yahoo's player_id**

### 2.3 Yahoo player_key (Composite)

**Verified fixture:** `tests/fixtures/yahoo_adp_injury.json`
```json
{
  "player_key": "469.p.7590",
  "player_id": "7590",
  "name": "Justin Verlander",
  "team": "DET",
  ...
}
```

**Format:** `{game_id}.p.{yahoo_player_id}`
- `469` = Yahoo game ID for MLB (not "mlb" sport prefix)
- `7590` = Proprietary Yahoo player ID (NOT mlbam_id)

**Code reference:** `backend/data_contracts/yahoo_player.py`
```python
class YahooPlayer(BaseModel):
    player_key: str   # Format: "{game_id}.p.{yahoo_id}" (e.g., "469.p.7590")
    player_id: str    # Yahoo proprietary ID (e.g., "7590") - NOT mlbam_id
    name: str
    team: str
    ...
```

**Key Observation:** Yahoo player_id is **proprietary** and requires name-based resolution to map to mlbam/BDL systems.

---

## 3. Identity Resolution Strategies

### Strategy A: Direct BDL-Yahoo Name Match (FALLBACK)

**Condition:** Use full_name matching between Yahoo and BDL (fragile, accents/Jr./Sr. issues).

**Resolution Path:**
```
Yahoo player_key → Yahoo name → normalize → match to BDL full_name
```

**Pros:**
- No external dependencies
- Fast if names match exactly

**Cons:**
- Name normalization complexity (accents, Jr./Sr., nicknames)
- Fragile — breaks on any name mismatch

### Strategy B: pybaseball Name → mlbam Mapping (RECOMMENDED)

**Condition:** Use pybaseball to resolve name → mlbam_id, then cross-reference.

**Resolution Path:**
```python
from pybaseball.playerid_lookup import playerid_lookup

# Map Yahoo player name → mlbam_id
lookup = playerid_lookup("Ohtani", "Shohei")
mlbam_id = lookup["key_mlbam"].iloc[0]  # 660271

# Then cross-reference mlbam_id with BDL (if BDL provides it)
```

**Existing code pattern:** `backend/fantasy_baseball/platoon_fetcher.py:123-133`
```python
from pybaseball.playerid_lookup import playerid_lookup
lookup = playerid_lookup(player_name.split()[0], player_name.split()[-1])
if not lookup.empty:
    return int(lookup.iloc[0]["key_fangraphs"])
```

**Pros:**
- Established library with comprehensive ID mapping
- Returns multiple ID types (mlbam, FanGraphs, etc.)

**Cons:**
- Name matching can fail (accents, Jr./Sr., nicknames)
- Network latency for lookups
- Requires fuzzy matching logic

### Strategy C: Cached Mapping Table (INSURANCE)

**Implementation:**
```sql
-- player_id_mapping table
CREATE TABLE player_id_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    yahoo_key TEXT UNIQUE NOT NULL,      -- e.g., "469.p.7590"
    yahoo_id TEXT NOT NULL,               -- e.g., "7590"
    mlbam_id INTEGER,                     -- Nullable until resolved
    bdl_id INTEGER,                       -- Nullable until resolved
    full_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,        -- For matching
    source TEXT NOT NULL DEFAULT 'manual', -- 'api', 'manual', 'pybaseball'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified DATE
);

-- Indexes for common lookups
CREATE INDEX idx_mapping_yahoo_key ON player_id_mapping(yahoo_key);
CREATE INDEX idx_mapping_mlbam ON player_id_mapping(mlbam_id);
CREATE INDEX idx_mapping_bdl ON player_id_mapping(bdl_id);
CREATE INDEX idx_mapping_normalized ON player_id_mapping(normalized_name);
```

**Use Case:**
- Cache successful pybaseball lookups
- Manual overrides for edge cases
- Audit trail for ID resolution failures

---

## 4. Recommended Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Yahoo Fantasy  │     │   ID Resolver    │     │   BDL Stats     │
│                 │     │   Service        │     │                 │
│  player_key:    │────▶│                  │◄────│  player.id:     │
│  "469.p.7590"   │     │  extract_name()  │     │  208            │
│                 │     │                  │     │                 │
│  name:          │────▶│  pybaseball      │     │  full_name:     │
│  "Justin        │     │  lookup          │◄────│  "Shohei        │
│   Verlander"    │     │                  │     │   Ohtani"       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  player_id_map   │
                        │  (cache table)   │
                        └──────────────────┘
```

### 4.1 Resolution Logic

```python
class PlayerIDResolver:
    """Resolves player identities across BDL, mlbam, and Yahoo systems."""
    
    def __init__(self, db_session, pybaseball_available: bool = True):
        self.db = db_session
        self.use_pybaseball = pybaseball_available
    
    def extract_yahoo_id(self, player_key: str) -> tuple[str, str]:
        """Extract game_id and yahoo_id from player_key.
        
        Args:
            player_key: Yahoo player_key (e.g., "469.p.7590")
            
        Returns:
            Tuple of (game_id, yahoo_id) -> ("469", "7590")
        """
        parts = player_key.split(".")
        if len(parts) == 3 and parts[1] == "p":
            return parts[0], parts[2]
        return "", ""
    
    def resolve_yahoo_to_mlbam(self, player_key: str, name: str) -> Optional[int]:
        """Resolve Yahoo player to mlbam ID via name lookup.
        
        Args:
            player_key: Yahoo player_key (e.g., "469.p.7590")
            name: Player full name from Yahoo
            
        Returns:
            mlbam_id if found, None otherwise
        """
        # Check cache first
        cached = self.db.query(IDMapping).filter_by(yahoo_key=player_key).first()
        if cached and cached.mlbam_id:
            return cached.mlbam_id
        
        # Try pybaseball name lookup
        if self.use_pybaseball:
            from pybaseball.playerid_lookup import playerid_lookup
            
            parts = name.split()
            if len(parts) >= 2:
                try:
                    lookup = playerid_lookup(parts[0], parts[-1])
                    if not lookup.empty:
                        mlbam_id = int(lookup.iloc[0]["key_mlbam"])
                        
                        # Cache the result
                        mapping = IDMapping(
                            yahoo_key=player_key,
                            yahoo_id=self.extract_yahoo_id(player_key)[1],
                            mlbam_id=mlbam_id,
                            full_name=name,
                            normalized_name=self._normalize_name(name),
                            source="pybaseball"
                        )
                        self.db.add(mapping)
                        self.db.commit()
                        return mlbam_id
                except Exception:
                    pass
        
        return None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching."""
        import unicodedata
        nfkd = unicodedata.normalize("NFKD", name.lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))
```

---

## 5. Decision Matrix

| Scenario | Approach | Resolution Strategy | Complexity |
|----------|----------|---------------------|------------|
| A | BDL provides mlbam_id | Direct join: mlbam_id → Yahoo (via pybaseball name) | Medium |
| B | BDL provides only internal ID | Name-based matching BDL ↔ Yahoo | Medium-High |
| C | Mixed coverage | Hybrid: cache + pybaseball fallback | High |

**Recommendation:** Implement Strategy B (pybaseball name mapping) with Strategy C (caching) as the foundation. Even if BDL provides mlbam_id, we'll still need name-based resolution for the initial mapping.

---

## 6. Critical Path Dependencies

### Blocked on K-A Completion

The resolution approach depends on whether BDL `/mlb/v1/stats` includes any external IDs:

**Next Steps:**
1. Execute schema verification from `reports/K_A_BDL_STATS_SPEC.md`
2. Determine if BDL stats endpoint returns:
   - mlbam_id (ideal)
   - Only BDL internal ID (requires name matching)
   - Both (best case)
3. Implement resolver accordingly

---

## 7. Database Schema (Proposed)

```sql
-- For caching and manual override
CREATE TABLE player_id_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    yahoo_key TEXT UNIQUE NOT NULL,       -- e.g., "469.p.7590"
    yahoo_id TEXT NOT NULL,                -- e.g., "7590"
    mlbam_id INTEGER,                      -- Nullable until resolved
    bdl_id INTEGER,                        -- Nullable until resolved
    full_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,         -- Lowercase, no accents
    source TEXT NOT NULL DEFAULT 'manual', -- 'pybaseball', 'manual', 'api'
    resolution_confidence FLOAT,           -- 0.0-1.0 for fuzzy matches
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified DATE
);

-- Indexes for common lookups
CREATE INDEX idx_mapping_yahoo_key ON player_id_mapping(yahoo_key);
CREATE INDEX idx_mapping_yahoo_id ON player_id_mapping(yahoo_id);
CREATE INDEX idx_mapping_mlbam ON player_id_mapping(mlbam_id);
CREATE INDEX idx_mapping_bdl ON player_id_mapping(bdl_id);
CREATE INDEX idx_mapping_normalized ON player_id_mapping(normalized_name);
```

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Name collisions in pybaseball | Medium | High | Use full name + team matching; manual override for conflicts |
| Yahoo player_key format varies | Low | High | Defensive parsing (handle both "469.p.XXX" and "mlb.p.XXX") |
| BDL names don't match Yahoo | Medium | **CRITICAL** | Implement fuzzy matching; cache manual corrections |
| Player trades (team mismatch) | Medium | Low | Team is secondary to name matching |
| Accented characters mismatch | High | Medium | Normalize both sides (é→e, ñ→n) before comparison |

---

## 9. Implementation Phases

### Phase 1: Schema Verification
- Execute BDL stats probe (K-A)
- Document which IDs are available from each source

### Phase 2: Name Normalization Library
- Implement `_normalize_name()` with full Unicode support
- Test against sample data from Yahoo and BDL

### Phase 3: Resolver Service
- Implement `PlayerIDResolver` class
- Add `player_id_mapping` table
- Wire pybaseball integration

### Phase 4: Integration
- Wire resolver into rolling window pipeline
- Add metrics/logging for resolution success rate
- Build admin UI for manual mapping corrections

---

## References

- `tests/fixtures/yahoo_adp_injury.json` — Verified Yahoo player_key format: `"469.p.7590"`
- `tests/fixtures/bdl_mlb_players.json` — Verified BDL player.id format: `208` (Ohtani)
- `backend/data_contracts/mlb_player.py` — BDL player contract (no mlbam_id)
- `backend/data_contracts/yahoo_player.py` — Yahoo player contract
- `backend/fantasy_baseball/statcast_scraper.py:147-154` — MLBAM ID lookup pattern
- `backend/fantasy_baseball/platoon_fetcher.py:123-133` — pybaseball lookup pattern
- `reports/YAHOO_FANTASY_API_RESEARCH.md` — "Yahoo uses proprietary player IDs"
