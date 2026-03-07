# A-26 Task 2: Seed-Spread Kelly Scalars — Implementation Spec

**Version:** 1.0  
**Date:** 2026-03-16  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**Target:** Claude Code (Master Architect)  
**Est. Implementation Time:** < 2 hours

---

## Executive Summary

This spec provides copy-paste ready code for implementing seed-spread Kelly scalars in the CBB Edge Analyzer V9 model. The implementation adds a fourth Kelly scalar (joining SNR, Integrity, and Portfolio) that reduces bet size when specific seed-spread combinations indicate market inefficiency.

**Three rules from K-1 research:**
1. **#5 seed favored by 6+ points** → 0.75× Kelly (33% ATS historical)
2. **#2 seed favored by 17+ points** → 0.75× Kelly (37% ATS historical)  
3. **#8 seed favored by ≤3 points** → 0.80× Kelly (underdog value trap)

---

## 1. BallDontLie API Contract

### Endpoint
```
GET https://api.balldontlie.io/ncaab/v1/march_madness_bracket?season={year}
```

### Authentication
- Header: `Authorization: {api_key}`
- Env var: `BALLDONTLIE_API_KEY`
- No Bearer prefix needed

### Response Shape
```json
{
  "data": [
    {
      "game_id": "401745970",
      "round": 1,
      "home_team": {
        "full_name": "Wisconsin Badgers",
        "seed": "3",
        "score": 85
      },
      "away_team": {
        "full_name": "Montana Grizzlies", 
        "seed": "14",
        "score": 66
      }
    }
  ]
}
```

### Key Fields
| Field | Type | Description |
|-------|------|-------------|
| `home_team.full_name` | string | Full team name (e.g., "Wisconsin Badgers") |
| `home_team.seed` | string | Seed number as string ("1" to "16") |
| `away_team.full_name` | string | Full team name |
| `away_team.seed` | string | Seed number as string |
| `round` | integer | 1=First Four, 2=First Round, etc. |

### Team Name Matching Strategy
Use fuzzy matching via existing `normalize_team_name()` from `team_mapping.py`:
1. Exact match first
2. Substring match (Odds API "Duke" → BallDontLie "Duke Blue Devils")
3. Return `None` if no match (graceful degradation)

---

## 2. tournament_data.py Blueprint

**File:** `backend/services/tournament_data.py`

```python
"""
Tournament seed data service — A-26 Task 2

Fetches NCAA March Madness bracket data including team seeds.
Primary source: BallDontLie API (paid, reliable)
Fallback: None (log warning and continue without seeds)

Seed-spread Kelly scalars are applied in betting_model.py based on
the seed data attached to game_dict by analysis.py.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import requests

from backend.services.team_mapping import normalize_team_name

logger = logging.getLogger(__name__)

_BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/ncaab/v1"
_CACHE: Dict = {}
_CACHE_TIMESTAMP: Optional[datetime] = None
_CACHE_TTL_HOURS = 6


class TournamentDataClient:
    """
    Client for fetching and caching NCAA tournament bracket data.
    
    Follows the same pattern as RatingsService in ratings.py:
    - Caches results for 6 hours (bracket doesn't change during tournament)
    - Fuzzy team name matching via normalize_team_name()
    - Graceful fallback to empty dict on API failure
    """
    
    def __init__(self):
        self.api_key = os.getenv("BALLDONTLIE_API_KEY")
        self._bracket_cache: Dict[str, int] = {}
        self._cache_timestamp: Optional[datetime] = None
    
    def fetch_bracket_data(self, season_year: Optional[int] = None) -> Dict[str, int]:
        """
        Fetch tournament bracket with seed data from BallDontLie API.
        
        Returns a dict mapping team_name -> seed (1-16).
        Returns empty dict if API key not set or API call fails.
        
        Args:
            season_year: Tournament season (e.g., 2026 for March 2026 tournament).
                        Defaults to current season from SEASON_YEAR env var.
        
        Returns:
            Dict[str, int]: {team_name: seed_number}
        """
        if not self.api_key:
            logger.debug("BALLDONTLIE_API_KEY not set — skipping seed fetch")
            return {}
        
        # Check cache
        if self._bracket_cache and self._cache_timestamp:
            age_hours = (datetime.utcnow() - self._cache_timestamp).total_seconds() / 3600
            if age_hours < _CACHE_TTL_HOURS:
                logger.debug("Using cached bracket data (%d teams)", len(self._bracket_cache))
                return self._bracket_cache
        
        year = season_year or int(os.getenv("SEASON_YEAR", datetime.utcnow().year))
        
        try:
            url = f"{_BALLDONTLIE_BASE_URL}/march_madness_bracket"
            headers = {"Authorization": self.api_key}
            params = {"season": year}
            
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            
            data = resp.json()
            seed_map = {}
            
            for game in data.get("data", []):
                home = game.get("home_team", {})
                away = game.get("away_team", {})
                
                home_name = home.get("full_name", "").strip()
                away_name = away.get("full_name", "").strip()
                
                if home_name and home.get("seed"):
                    try:
                        seed_map[home_name] = int(home["seed"])
                    except (ValueError, TypeError):
                        pass
                
                if away_name and away.get("seed"):
                    try:
                        seed_map[away_name] = int(away["seed"])
                    except (ValueError, TypeError):
                        pass
            
            self._bracket_cache = seed_map
            self._cache_timestamp = datetime.utcnow()
            
            logger.info("TournamentDataClient: loaded %d teams from BallDontLie", len(seed_map))
            return seed_map
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("BallDontLie bracket not available yet (404)")
            else:
                logger.warning("BallDontLie API error: %s", e)
            return {}
        except Exception as e:
            logger.warning("Failed to fetch tournament bracket: %s", e)
            return {}
    
    def get_team_seed(
        self, 
        team_name: str, 
        bracket_data: Optional[Dict[str, int]] = None
    ) -> Optional[int]:
        """
        Look up seed for a team using fuzzy name matching.
        
        Args:
            team_name: Team name from Odds API (e.g., "Duke Blue Devils")
            bracket_data: Pre-fetched bracket dict. If None, fetches fresh.
        
        Returns:
            int: Seed number (1-16) or None if not found
        """
        if bracket_data is None:
            bracket_data = self.fetch_bracket_data()
        
        if not bracket_data:
            return None
        
        # Try exact match first
        if team_name in bracket_data:
            return bracket_data[team_name]
        
        # Try fuzzy matching via normalize_team_name
        normalized = normalize_team_name(team_name, list(bracket_data.keys()))
        if normalized:
            return bracket_data.get(normalized)
        
        # Substring fallback (e.g., "Duke" matches "Duke Blue Devils")
        team_lower = team_name.lower()
        for full_name, seed in bracket_data.items():
            if team_lower in full_name.lower() or full_name.lower() in team_lower:
                return seed
        
        return None
    
    def get_game_seeds(
        self, 
        home_team: str, 
        away_team: str,
        bracket_data: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Get seeds for both teams in a game.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            bracket_data: Pre-fetched bracket dict
        
        Returns:
            Tuple[Optional[int], Optional[int]]: (home_seed, away_seed)
        """
        if bracket_data is None:
            bracket_data = self.fetch_bracket_data()
        
        home_seed = self.get_team_seed(home_team, bracket_data)
        away_seed = self.get_team_seed(away_team, bracket_data)
        
        return home_seed, away_seed


# ---------------------------------------------------------------------------
# Singleton instance (follows ratings.py pattern)
# ---------------------------------------------------------------------------
_tournament_client: Optional[TournamentDataClient] = None


def get_tournament_client() -> TournamentDataClient:
    """Return singleton TournamentDataClient instance."""
    global _tournament_client
    if _tournament_client is None:
        _tournament_client = TournamentDataClient()
    return _tournament_client


def fetch_tournament_bracket(season_year: Optional[int] = None) -> Dict[str, int]:
    """Convenience function — fetch bracket via singleton client."""
    return get_tournament_client().fetch_bracket_data(season_year)


def get_team_seed(team_name: str, bracket_data: Optional[Dict[str, int]] = None) -> Optional[int]:
    """Convenience function — get seed via singleton client."""
    return get_tournament_client().get_team_seed(team_name, bracket_data)
```

---

## 3. analysis.py Enrichment

### Location: Inside `run_nightly_analysis()`, Pass 2 loop

**Insert the following code block at line ~1017** (after `game_input` dict is built):

```python
                    # ---- Seed data enrichment (A-26 Task 2) ----------------
                    # Only fetch bracket once per analysis run
                    if 'tournament_bracket' not in locals():
                        from backend.services.tournament_data import fetch_tournament_bracket
                        tournament_bracket = fetch_tournament_bracket()
                    
                    home_seed = None
                    away_seed = None
                    if tournament_bracket:
                        from backend.services.tournament_data import get_tournament_client
                        home_seed, away_seed = get_tournament_client().get_game_seeds(
                            home_team, away_team, tournament_bracket
                        )
                        if home_seed or away_seed:
                            logger.debug(
                                "Seeds for %s @ %s: home=%s, away=%s",
                                away_team, home_team, home_seed, away_seed
                            )
                    
                    # Attach seeds to game_input for betting model
                    game_input["home_seed"] = home_seed
                    game_input["away_seed"] = away_seed
```

### Exact Insertion Point

Find this existing block in `analysis.py` (lines ~1017-1021):
```python
                    game_input = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "is_neutral": game_data.get("is_neutral", False),
                    }
```

**Replace with:**
```python
                    game_input = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "is_neutral": game_data.get("is_neutral", False),
                    }
                    
                    # ---- Seed data enrichment (A-26 Task 2) ----------------
                    # Only fetch bracket once per analysis run
                    if 'tournament_bracket' not in locals():
                        from backend.services.tournament_data import fetch_tournament_bracket
                        tournament_bracket = fetch_tournament_bracket()
                    
                    home_seed = None
                    away_seed = None
                    if tournament_bracket:
                        from backend.services.tournament_data import get_tournament_client
                        home_seed, away_seed = get_tournament_client().get_game_seeds(
                            home_team, away_team, tournament_bracket
                        )
                        if home_seed or away_seed:
                            logger.debug(
                                "Seeds for %s @ %s: home=%s, away=%s",
                                away_team, home_team, home_seed, away_seed
                            )
                    
                    # Attach seeds to game_input for betting model
                    game_input["home_seed"] = home_seed
                    game_input["away_seed"] = away_seed
```

---

## 4. betting_model.py Scalar Implementation

### 4.1 Add `_seed_spread_kelly_scalar()` method

**Location:** After `_integrity_kelly_scalar()` method (around line 497)

**Insert this method:**

```python
    def _seed_spread_kelly_scalar(
        self,
        home_seed: Optional[int],
        away_seed: Optional[int],
        spread: Optional[float],
    ) -> float:
        """
        Calculate Kelly multiplier based on seed-spread combinations.
        
        K-1 research identified three seed-spread market inefficiencies:
        
        Rule 1: #5 seed favored by 6+ points → 0.75×
            - Historical: 33% ATS (significantly underperforms market expectation)
            - Psychology: Public overvalues "proven" mid-major favorites
            
        Rule 2: #2 seed favored by 17+ points → 0.75×
            - Historical: 37% ATS (market overvalues elite vs 15-seed mismatch)
            - Context: Large spreads assume blowout; tournament variance is higher
            
        Rule 3: #8 seed favored by ≤3 points → 0.80×
            - Historical: Underdog value trap (8-seeds often overvalued vs 9-seeds)
            - Context: 8/9 games are effectively coin flips; small favorite = warning
        
        Scalars are overridable via environment variables:
            SEED5_FAV6PLUS_SCALAR (default 0.75)
            SEED2_FAV17PLUS_SCALAR (default 0.75)
            SEED8_FAV3MINUS_SCALAR (default 0.80)
        
        Args:
            home_seed: Home team tournament seed (1-16) or None
            away_seed: Away team tournament seed (1-16) or None
            spread: Home team spread (negative = home favored)
        
        Returns:
            Kelly multiplier ∈ (0, 1.0]. Returns 1.0 if:
                - Either seed is None (not tournament or data unavailable)
                - No seed-spread rule applies
        """
        # Skip if seed data unavailable
        if home_seed is None or away_seed is None:
            return 1.0
        
        # Skip if spread unavailable
        if spread is None:
            return 1.0
        
        # Determine which team is favored
        if spread < 0:
            favored_seed = home_seed
            spread_abs = abs(spread)
        elif spread > 0:
            favored_seed = away_seed
            spread_abs = abs(spread)
        else:
            # Pick'em — no favorite
            return 1.0
        
        # Apply seed-spread rules
        if favored_seed == 5 and spread_abs >= 6.0:
            return get_float_env("SEED5_FAV6PLUS_SCALAR", "0.75")
        
        if favored_seed == 2 and spread_abs >= 17.0:
            return get_float_env("SEED2_FAV17PLUS_SCALAR", "0.75")
        
        if favored_seed == 8 and spread_abs <= 3.0:
            return get_float_env("SEED8_FAV3MINUS_SCALAR", "0.80")
        
        return 1.0
```

### 4.2 Apply Scalar in Kelly Chain

**Location:** In `analyze_game()` method, after V9 scalars (after line 1510)

**Find this existing block:**
```python
        kelly_frac *= v9_scalar

        # ================================================================
        # V9 INTEGRITY ABORT GATE
```

**Replace with:**
```python
        kelly_frac *= v9_scalar
        
        # ================================================================
        # A-26: Seed-Spread Kelly Scalar (Tournament Only)
        # ================================================================
        home_seed = game_data.get("home_seed")
        away_seed = game_data.get("away_seed")
        spread = odds.get("spread")
        
        seed_spread_scalar = self._seed_spread_kelly_scalar(
            home_seed, away_seed, spread
        )
        if seed_spread_scalar < 1.0:
            kelly_frac *= seed_spread_scalar
            notes.append(
                f"Seed-Spread scalar: {seed_spread_scalar:.2f}× "
                f"(favored_seed={home_seed if spread < 0 else away_seed}, "
                f"spread={spread:+.1f})"
            )
            logger.debug(
                "Seed-Spread scalar applied: %.2f× (home_seed=%s, away_seed=%s, spread=%.1f)",
                seed_spread_scalar, home_seed, away_seed, spread
            )

        # ================================================================
        # V9 INTEGRITY ABORT GATE
```

### 4.3 Persist Scalar in full_analysis Output

**Location:** In the `full_analysis_dict` construction (around line 1791)

**Find this existing block:**
```python
            'calculations': {
                ...
                'integrity_verdict': integrity_verdict,
                'integrity_kelly_scalar': integrity_scalar,
                ...
            },
```

**Add after `'integrity_kelly_scalar'`: line:**
```python
                'seed_spread_scalar': seed_spread_scalar,
                'home_seed': home_seed,
                'away_seed': away_seed,
```

---

## 5. Final Scalar Table

| Seed | Spread Condition | Scalar | Historical ATS | Rationale |
|------|-----------------|--------|----------------|-----------|
| **#5** | Favored by ≥6 pts | **0.75×** | 33% ATS | Public overvalues mid-major favorites |
| **#2** | Favored by ≥17 pts | **0.75×** | 37% ATS | Market overvalues blowout expectation |
| **#8** | Favored by ≤3 pts | **0.80×** | Underdog trap | 8/9 games = coin flip; avoid small fav |

**Default scalar (no match):** 1.0× (no adjustment)

**Override env vars:**
- `SEED5_FAV6PLUS_SCALAR` (default: 0.75)
- `SEED2_FAV17PLUS_SCALAR` (default: 0.75)
- `SEED8_FAV3MINUS_SCALAR` (default: 0.80)

---

## 6. Environment Variables

Add to `.env` and Railway:

```bash
# ==========================================
# A-26: Seed-Spread Kelly Scalars
# ==========================================

# BallDontLie API (required for tournament seed data)
BALLDONTLIE_API_KEY=your_api_key_here

# Seed-spread scalar overrides (optional)
SEED5_FAV6PLUS_SCALAR=0.75    # #5 seed fav 6+ pts
SEED2_FAV17PLUS_SCALAR=0.75   # #2 seed fav 17+ pts  
SEED8_FAV3MINUS_SCALAR=0.80   # #8 seed fav ≤3 pts
```

**Railway setup:**
1. Add `BALLDONTLIE_API_KEY` to Railway Variables
2. Scalars are optional — defaults are production-ready

---

## 7. Test Specifications

### Test File: `tests/test_seed_spread_scalars.py`

```python
"""Tests for A-26 Seed-Spread Kelly Scalars."""

import pytest
from unittest.mock import patch

from backend.betting_model import CBBEdgeModel


class TestSeedSpreadScalars:
    """Test seed-spread Kelly scalar calculation."""
    
    def setup_method(self):
        """Initialize model for each test."""
        self.model = CBBEdgeModel()
    
    # ------------------------------------------------------------------
    # Rule 1: #5 seed favored by 6+ points → 0.75×
    # ------------------------------------------------------------------
    
    def test_seed5_favored_6_plus_home(self):
        """Home team is #5 seed favored by 6.5 → 0.75×"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=-6.5
        )
        assert result == 0.75
    
    def test_seed5_favored_6_plus_away(self):
        """Away team is #5 seed favored by 7 → 0.75×"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=12, away_seed=5, spread=7.0
        )
        assert result == 0.75
    
    def test_seed5_favored_5_points_no_scalar(self):
        """#5 seed favored by only 5 → no scalar (1.0×)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=-5.0
        )
        assert result == 1.0
    
    # ------------------------------------------------------------------
    # Rule 2: #2 seed favored by 17+ points → 0.75×
    # ------------------------------------------------------------------
    
    def test_seed2_favored_17_plus_home(self):
        """Home team is #2 seed favored by 17.5 → 0.75×"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=2, away_seed=15, spread=-17.5
        )
        assert result == 0.75
    
    def test_seed2_favored_17_plus_away(self):
        """Away team is #2 seed favored by 18 → 0.75×"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=15, away_seed=2, spread=18.0
        )
        assert result == 0.75
    
    def test_seed2_favored_16_points_no_scalar(self):
        """#2 seed favored by only 16 → no scalar (1.0×)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=2, away_seed=15, spread=-16.0
        )
        assert result == 1.0
    
    # ------------------------------------------------------------------
    # Rule 3: #8 seed favored by ≤3 points → 0.80×
    # ------------------------------------------------------------------
    
    def test_seed8_favored_3_points_home(self):
        """Home team is #8 seed favored by 3 → 0.80×"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=8, away_seed=9, spread=-3.0
        )
        assert result == 0.80
    
    def test_seed8_favored_2_points_away(self):
        """Away team is #8 seed favored by 2.5 → 0.80×"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=9, away_seed=8, spread=2.5
        )
        assert result == 0.80
    
    def test_seed8_favored_4_points_no_scalar(self):
        """#8 seed favored by 4 → no scalar (1.0×)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=8, away_seed=9, spread=-4.0
        )
        assert result == 1.0
    
    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    
    def test_no_seed_data(self):
        """Missing seed data → 1.0× (no scalar)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=None, away_seed=5, spread=-6.5
        )
        assert result == 1.0
    
    def test_no_spread_data(self):
        """Missing spread → 1.0× (no scalar)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=None
        )
        assert result == 1.0
    
    def test_pick_em_no_favorite(self):
        """Pick'em spread (0) → 1.0× (no scalar)"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=0.0
        )
        assert result == 1.0
    
    def test_underdog_seed_no_scalar(self):
        """Seed conditions only apply to FAVORITE"""
        # #5 seed as underdog (getting points) → no scalar
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=4, spread=6.5  # Home is underdog
        )
        assert result == 1.0
    
    # ------------------------------------------------------------------
    # Env var overrides
    # ------------------------------------------------------------------
    
    @patch.dict('os.environ', {'SEED5_FAV6PLUS_SCALAR': '0.60'})
    def test_env_override_seed5(self):
        """SEED5_FAV6PLUS_SCALAR env var overrides default"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=5, away_seed=12, spread=-6.5
        )
        assert result == 0.60
    
    @patch.dict('os.environ', {'SEED2_FAV17PLUS_SCALAR': '0.50'})
    def test_env_override_seed2(self):
        """SEED2_FAV17PLUS_SCALAR env var overrides default"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=2, away_seed=15, spread=-17.5
        )
        assert result == 0.50
    
    @patch.dict('os.environ', {'SEED8_FAV3MINUS_SCALAR': '0.70'})
    def test_env_override_seed8(self):
        """SEED8_FAV3MINUS_SCALAR env var overrides default"""
        result = self.model._seed_spread_kelly_scalar(
            home_seed=8, away_seed=9, spread=-3.0
        )
        assert result == 0.70
```

### Test File: `tests/test_tournament_data.py`

```python
"""Tests for tournament_data.py — A-26 Task 2."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.services.tournament_data import (
    TournamentDataClient,
    get_tournament_client,
    fetch_tournament_bracket,
)


class TestTournamentDataClient:
    """Test TournamentDataClient functionality."""
    
    def setup_method(self):
        """Fresh client for each test."""
        self.client = TournamentDataClient()
        self.client._bracket_cache = {}
        self.client._cache_timestamp = None
    
    @patch('backend.services.tournament_data.requests.get')
    @patch.dict('os.environ', {'BALLDONTLIE_API_KEY': 'test_key'})
    def test_fetch_bracket_success(self, mock_get):
        """Successful API call returns seed dict."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "home_team": {"full_name": "Duke Blue Devils", "seed": "1"},
                    "away_team": {"full_name": "Vermont Catamounts", "seed": "16"},
                },
                {
                    "home_team": {"full_name": "Kentucky Wildcats", "seed": "3"},
                    "away_team": {"full_name": "Oakland Golden Grizzlies", "seed": "14"},
                }
            ]
        }
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp
        
        result = self.client.fetch_bracket_data(season_year=2026)
        
        assert result["Duke Blue Devils"] == 1
        assert result["Vermont Catamounts"] == 16
        assert result["Kentucky Wildcats"] == 3
        assert result["Oakland Golden Grizzlies"] == 14
        assert len(result) == 4
    
    @patch.dict('os.environ', {}, clear=True)
    def test_no_api_key_returns_empty(self):
        """Missing API key returns empty dict gracefully."""
        result = self.client.fetch_bracket_data()
        assert result == {}
    
    @patch('backend.services.tournament_data.requests.get')
    @patch.dict('os.environ', {'BALLDONTLIE_API_KEY': 'test_key'})
    def test_api_404_returns_empty(self, mock_get):
        """404 error (bracket not ready) returns empty dict."""
        from requests import HTTPError
        mock_resp = Mock()
        mock_resp.raise_for_status.side_effect = HTTPError(response=Mock(status_code=404))
        mock_get.return_value = mock_resp
        
        result = self.client.fetch_bracket_data()
        assert result == {}
    
    def test_caching(self):
        """Bracket data is cached for 6 hours."""
        self.client._bracket_cache = {"Duke Blue Devils": 1}
        self.client._cache_timestamp = datetime.utcnow()
        
        # Should return cached data without API call
        result = self.client.fetch_bracket_data()
        assert result == {"Duke Blue Devils": 1}
    
    def test_cache_expired(self):
        """Cache older than 6 hours triggers refetch."""
        from datetime import timedelta
        self.client._bracket_cache = {"Old Data": 1}
        self.client._cache_timestamp = datetime.utcnow() - timedelta(hours=7)
        
        # No API key, so should return empty after trying to refresh
        result = self.client.fetch_bracket_data()
        assert result == {}
    
    def test_get_team_seed_exact_match(self):
        """Exact team name match returns seed."""
        bracket = {"Duke Blue Devils": 1, "Vermont Catamounts": 16}
        result = self.client.get_team_seed("Duke Blue Devils", bracket)
        assert result == 1
    
    def test_get_team_seed_substring_match(self):
        """Substring match works for partial names."""
        bracket = {"Duke Blue Devils": 1}
        result = self.client.get_team_seed("Duke", bracket)
        assert result == 1
    
    def test_get_team_seed_not_found(self):
        """Unknown team returns None."""
        bracket = {"Duke Blue Devils": 1}
        result = self.client.get_team_seed("Unknown Team", bracket)
        assert result is None
    
    def test_get_game_seeds(self):
        """Get seeds for both teams."""
        bracket = {"Duke Blue Devils": 1, "Vermont Catamounts": 16}
        home, away = self.client.get_game_seeds(
            "Duke Blue Devils", "Vermont Catamounts", bracket
        )
        assert home == 1
        assert away == 16


class TestTournamentDataSingleton:
    """Test singleton pattern."""
    
    def test_singleton_returns_same_instance(self):
        """get_tournament_client() returns same instance."""
        client1 = get_tournament_client()
        client2 = get_tournament_client()
        assert client1 is client2
```

---

## 8. Implementation Checklist

### Phase 1: New Module (20 min)
- [ ] Create `backend/services/tournament_data.py` with code above
- [ ] Add `BALLDONTLIE_API_KEY` to `.env.example`

### Phase 2: analysis.py Integration (15 min)
- [ ] Insert seed enrichment block after line ~1017 in `run_nightly_analysis()`

### Phase 3: betting_model.py Integration (30 min)
- [ ] Add `_seed_spread_kelly_scalar()` method after `_integrity_kelly_scalar()`
- [ ] Apply scalar in Kelly chain after V9 scalars
- [ ] Add scalar to `full_analysis_dict` output

### Phase 4: Tests (30 min)
- [ ] Create `tests/test_seed_spread_scalars.py` with 14 test cases
- [ ] Create `tests/test_tournament_data.py` with client tests
- [ ] Run `pytest tests/test_seed_spread_scalars.py tests/test_tournament_data.py -v`

### Phase 5: Railway Deployment (15 min)
- [ ] Add `BALLDONTLIE_API_KEY` to Railway environment variables
- [ ] Deploy and verify bracket fetch in logs

---

## 9. Verification Steps

### Local Testing
```bash
# Test tournament data client
python -c "
from backend.services.tournament_data import fetch_tournament_bracket
bracket = fetch_tournament_bracket(2026)
print(f'Loaded {len(bracket)} teams')
if bracket:
    for team, seed in list(bracket.items())[:3]:
        print(f'  {team}: #{seed}')
"

# Run tests
pytest tests/test_seed_spread_scalars.py -v
pytest tests/test_tournament_data.py -v
```

### Production Verification
1. Check logs for: `"TournamentDataClient: loaded X teams from BallDontLie"`
2. Check logs for: `"Seeds for {away} @ {home}: home={seed}, away={seed}"`
3. Check logs for: `"Seed-Spread scalar: 0.75× ..."`
4. Verify `full_analysis.calculations.seed_spread_scalar` in DB predictions

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| BallDontLie API down | Returns empty dict → model runs without seed scalars (graceful degradation) |
| Team name mismatch | Fuzzy matching + substring fallback → None if still no match |
| Cache stale | 6-hour TTL ensures fresh data daily |
| Wrong seed applied | Clear logging of seed + spread + scalar in notes |

---

*Spec complete — ready for implementation by Claude Code*
