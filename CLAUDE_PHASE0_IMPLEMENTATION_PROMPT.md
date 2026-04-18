# Phase 0: Stat Contract Infrastructure + UI Contracts

> **Purpose:** Build the validated, self-describing stat contract system that every backend service and frontend page derives from. Then define the Pydantic UI contracts that reference it. This is the foundation — all 9 phases depend on these contracts being correct and complete.
>
> **Session type:** Code implementation (not research or planning)
> **Key principle:** The JSON contract is the single source of truth. Pydantic models validate it. Backend services load it at startup. TypeScript types are generated from Pydantic via OpenAPI. The frontend **never hand-codes category names**.
>
> **Constraints:** No new API endpoints, no data pipelines, no DB migrations. Type definitions, validation, and a contract loader only.

---

## 1. READ FIRST (mandatory, in order)

1. `HANDOFF.md` — sections: "UI Contract Authority", "Active Workstream: Phase 0", "Gate 0 Criteria"
2. `reports/2026-04-17-ui-specification-contract-audit.md` — sections 1A–1D (field IDs GH-*, MS-*, PR-*)
3. `backend/stat_contract/fantasy_stat_contract.json` — **the v2 JSON contract (already staged)**. This is the authoritative source of all stat definitions, scoring categories, weekly rules, external ID mappings, and display order. Read it in full.
4. `Examples/contract_schema.py` — Pydantic v2 models that validate the JSON. **This is your primary reference** for the schema implementation.
5. `Examples/master_stat_registry.py` — Python dataclass stat definitions. **This is your primary reference** for the registry implementation.
6. `Examples/contract_builder.py` — Pure function that builds a validated contract from Yahoo league settings. Reference for builder implementation.
7. `backend/contracts.py` — existing decision contracts. UI contracts go HERE after `ExecutionDecision`.
8. `backend/utils/fantasy_stat_contract.py` — the **old** v1 loader. Shows what currently exists and what consumers import. The new loader replaces this.

---

## 2. ARCHITECTURAL VISION

```
┌─────────────────────────────────────────────────────────┐
│              Yahoo League Settings API                  │
│              (fetched once per season)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│      contract_builder.py  (pure function)               │
│      Yahoo settings dict → validated FantasyStatContract│
└──────────────────────┬──────────────────────────────────┘
                       │ writes
                       ▼
┌─────────────────────────────────────────────────────────┐
│   fantasy_stat_contract.json  (file on disk)            │
│   Authoritative. Cacheable. League-specific.            │
│   Contains: stats, supporting_stats, scoring_categories,│
│   weekly_rules, yahoo_id_index, matchup_display_order   │
└──────────────────────┬──────────────────────────────────┘
                       │ loaded at startup
                       ▼
┌─────────────────────────────────────────────────────────┐
│   schema.py → FantasyStatContract (Pydantic v2)         │
│   Validates JSON. Catches schema drift. Type-safe.      │
│   Every backend service imports the validated singleton. │
└──────────────────────┬──────────────────────────────────┘
                       │ derived constants
                       ▼
┌─────────────────────────────────────────────────────────┐
│   UI Contracts (backend/contracts.py)                   │
│   MatchupScoreboardRow, CanonicalPlayerRow, etc.        │
│   Category fields keyed by canonical codes from contract│
│   Frontend TypeScript types generated via OpenAPI export │
└─────────────────────────────────────────────────────────┘
```

**The flow is:** Yahoo → Builder → JSON → Schema validates → Loader exposes singleton → Backend services + UI contracts reference it → OpenAPI → TypeScript types → Frontend

---

## 3. CANONICAL CODE MAPPING (old → new)

The v1 JSON used ambiguous display-style keys. The v2 JSON uses **disambiguated canonical codes**:

| Old Key (v1) | New Canonical Code (v2) | Why |
|---|---|---|
| `HR` | `HR_B` | Disambiguate hitter HR from pitcher HR allowed |
| `HRA` | `HR_P` | Pitcher home runs allowed (canonical, not display) |
| `K(B)` | `K_B` | No parens in identifiers — batter strikeouts |
| `K` | `K_P` | Pitcher strikeouts (disambiguated from batter K) |
| `K/9` | `K_9` | No slash in identifiers |

**Unchanged:** `R`, `H`, `RBI`, `TB`, `AVG`, `OPS`, `NSB`, `W`, `L`, `ERA`, `WHIP`, `QS`, `NSV`

**The 18 scoring categories (canonical codes):**
- **Batting (9):** `R`, `H`, `HR_B`, `RBI`, `K_B`, `TB`, `AVG`, `OPS`, `NSB`
- **Pitching (9):** `W`, `L`, `HR_P`, `K_P`, `ERA`, `WHIP`, `K_9`, `QS`, `NSV`
- **Lower-is-better (5):** `ERA`, `WHIP`, `L`, `K_B`, `HR_P`

**Display-only stats** (not scoring, but appear on matchup views): `IP`, `GS`, `H_AB`

**Supporting stats** (atomic inputs for aggregation): `AB`, `BB`, `HBP`, `SF`, `SB`, `CS`, `OBP`, `SLG`, `ER`, `IP_OUTS`, `BB_P`, `H_P`, `SV`, `BS`, `HLD`

Each stat entry in the JSON carries its own `short_label` (the display string the UI shows). The canonical code is the internal identifier. The `short_label` is the human-facing label. Example: canonical `HR_B` has `short_label: "HR"`, canonical `K_B` has `short_label: "K"`.

---

## 4. FILE PLACEMENT

All new code goes in the `backend/stat_contract/` package. The JSON is already staged there.

```
backend/stat_contract/
├── __init__.py                      # Exports CONTRACT singleton + convenience constants
├── fantasy_stat_contract.json       # ✅ ALREADY STAGED — the v2 JSON
├── schema.py                        # Pydantic v2 models that validate the JSON
├── registry.py                      # Frozen dataclass stat definitions (Python knowledge base)
├── builder.py                       # Pure function: Yahoo settings dict → FantasyStatContract
└── loader.py                        # Loads JSON from disk, validates, returns typed contract

backend/contracts.py                 # EXISTING — append UI contracts after ExecutionDecision

tests/test_stat_contract_schema.py   # Schema validation tests
tests/test_ui_contracts.py           # UI contract shape tests
```

---

## 5. IMPLEMENTATION — DELIVERABLE BY DELIVERABLE

### Deliverable D1: `backend/stat_contract/schema.py`

Adapt `Examples/contract_schema.py` into production code. This file defines the Pydantic v2 models that validate the JSON.

**Key models:**
- `Aggregation` — aggregation method + parameters
- `ExternalIds` — Yahoo stat_id, MLB Stats API, BDL, pybaseball mappings
- `StatEntry` — full stat definition (for `stats` section)
- `SupportingStatEntry` — simplified stat definition (for `supporting_stats` section — fewer required fields)
- `ScoringCategories` — batting/pitching lists + total_count + win_threshold
- `WeeklyRules` — IP minimum, acquisitions, waiver window
- `Provenance` — when/how the contract was generated
- `FantasyStatContract` — the root model that holds everything

**Critical details:**
- Use Pydantic v2 style: `model_config = ConfigDict(frozen=True, extra="forbid")`
- The `supporting_stats` section in the JSON has a **simplified schema** (no `canonical_code`, `is_scoring_category`, `scoring_role`, `aggregation`, `direction`, `valid_range`). Create a `SupportingStatEntry` model with only the fields present: `display_label`, `scope`, `data_type`, `precision`, `external_ids`, and optional `notes`.
- `StatEntry` must validate that `valid_range` has exactly 2 elements (when not null)
- `ScoringCategories` must validate that `total_count == len(batting) + len(pitching)`
- `WeeklyRules` must validate that `pitcher_ip_minimum_outs == round(pitcher_ip_minimum * 3)`
- `FantasyStatContract` must cross-validate: every code in `scoring_categories.batting + scoring_categories.pitching` must exist in `stats` with `is_scoring_category: true`
- `FantasyStatContract` must validate: every code in `yahoo_id_index` values must exist in `stats` or `supporting_stats`

### Deliverable D2: `backend/stat_contract/registry.py`

Adapt `Examples/master_stat_registry.py` into production code. This is the Python-side knowledge base — frozen dataclasses for every stat the platform understands.

**Key contents:**
- `Aggregation`, `ExternalIds`, `StatDefinition` — frozen dataclasses
- `BATTING_STATS: dict[str, StatDefinition]` — 9 batting scoring stats
- `PITCHING_STATS: dict[str, StatDefinition]` — 9 pitching scoring stats
- `DISPLAY_STATS: dict[str, StatDefinition]` — IP, GS, H_AB
- `SUPPORTING_STATS: dict[str, StatDefinition]` — AB, BB, HBP, SF, SB, CS, OBP, SLG, ER, IP_OUTS, BB_P, H_P, SV, BS, HLD
- `ALL_STATS: dict[str, StatDefinition]` — union of all the above
- `YAHOO_ID_TO_CANONICAL: dict[int, str]` — built from ALL_STATS external_ids

Copy the complete stat definitions from the example — every field, every note, every external ID. Do NOT abbreviate. The registry must match the JSON exactly.

### Deliverable D3: `backend/stat_contract/builder.py`

Adapt `Examples/contract_builder.py` into production code. Pure function that takes a Yahoo league settings dict and produces a validated `FantasyStatContract`.

**Key function:**
```python
def build_contract(
    *,
    yahoo_settings: dict[str, Any],
    league_id: str,
    season: int,
    now: datetime | None = None,
) -> FantasyStatContract:
```

**Key behaviors:**
- Extracts scored stat IDs from Yahoo settings response
- Maps Yahoo stat IDs to canonical codes via the registry
- Raises `UnknownYahooStatError` if Yahoo returns an unmapped stat_id
- Builds `StatEntry` objects from registry definitions
- Only includes supporting stats that are referenced by scored stats' aggregation
- Builds `yahoo_id_index` covering primary and alt Yahoo IDs
- Produces a fully validated `FantasyStatContract` instance

Include the helper functions from the example: `_extract_scored_stat_ids`, `_extract_league_key`, `_extract_format`, `_extract_win_threshold`, `_extract_weekly_rules`, `_to_entry`, `_collect_referenced_supporting_stats`, `_build_yahoo_id_index`, `_build_matchup_display_order`.

### Deliverable D4: `backend/stat_contract/loader.py`

New file — loads the JSON from disk and validates it against the schema.

```python
"""Load and validate the fantasy stat contract at startup."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .schema import FantasyStatContract

_CONTRACT_PATH = Path(__file__).parent / "fantasy_stat_contract.json"


@lru_cache(maxsize=1)
def load_contract() -> FantasyStatContract:
    """Load and validate the stat contract. Cached — call freely."""
    if not _CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Stat contract not found at {_CONTRACT_PATH}. "
            "Run the contract generator or place the JSON manually."
        )
    raw = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    return FantasyStatContract.model_validate(raw)
```

### Deliverable D5: `backend/stat_contract/__init__.py`

Exports the loaded contract singleton and convenience constants derived from it.

```python
"""Stat contract package — the single source of truth for all stat semantics."""

from .loader import load_contract
from .schema import (
    FantasyStatContract,
    StatEntry,
    SupportingStatEntry,
    ScoringCategories,
    WeeklyRules,
    Aggregation,
    ExternalIds,
    Provenance,
)

# Validated contract singleton — loaded once at import time
CONTRACT: FantasyStatContract = load_contract()

# Convenience constants derived from the loaded contract
SCORING_CATEGORY_CODES: frozenset[str] = frozenset(
    CONTRACT.scoring_categories.batting + CONTRACT.scoring_categories.pitching
)
BATTING_CODES: frozenset[str] = frozenset(CONTRACT.scoring_categories.batting)
PITCHING_CODES: frozenset[str] = frozenset(CONTRACT.scoring_categories.pitching)
LOWER_IS_BETTER: frozenset[str] = frozenset(
    code for code, entry in CONTRACT.stats.items()
    if entry.is_scoring_category and entry.direction == "lower_is_better"
)
YAHOO_ID_INDEX: dict[str, str] = dict(CONTRACT.yahoo_id_index)
MATCHUP_DISPLAY_ORDER: list[str] = list(CONTRACT.matchup_display_order)

__all__ = [
    "CONTRACT",
    "SCORING_CATEGORY_CODES",
    "BATTING_CODES",
    "PITCHING_CODES",
    "LOWER_IS_BETTER",
    "YAHOO_ID_INDEX",
    "MATCHUP_DISPLAY_ORDER",
    "load_contract",
    "FantasyStatContract",
    "StatEntry",
    "SupportingStatEntry",
    "ScoringCategories",
    "WeeklyRules",
]
```

### Deliverable D6: UI contracts in `backend/contracts.py`

Append the following AFTER the existing `ExecutionDecision` class. Use the same frozen-config style as existing contracts. **All category-keyed dicts must use canonical codes from the loaded contract, NOT hardcoded strings.**

Add this import at the top of the file:
```python
from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER, BATTING_CODES
```

#### P0-1: CategoryStatusTag

```python
class CategoryStatusTag(str, Enum):
    """Classification of a scoring category's matchup status.

    Threshold definitions (for L3/L4 classification logic):
    - LOCKED_WIN:   Monte Carlo win probability > 90%
    - LOCKED_LOSS:  Monte Carlo win probability < 10%
    - LEANING_WIN:  65% < win probability <= 90%
    - LEANING_LOSS: 10% <= win probability < 35%
    - BUBBLE:       35% <= win probability <= 65%
    """
    LOCKED_WIN = "locked_win"
    LOCKED_LOSS = "locked_loss"
    LEANING_WIN = "leaning_win"
    LEANING_LOSS = "leaning_loss"
    BUBBLE = "bubble"
```

#### P0-2: IPPaceFlag + ConstraintBudget

```python
class IPPaceFlag(str, Enum):
    """Weekly innings pitched pace relative to league minimum."""
    BEHIND = "behind"
    ON_TRACK = "on_track"
    AHEAD = "ahead"

class ConstraintBudget(BaseModel):
    """Current constraint state for the global header. Fields map to GH-6 through GH-14."""
    acquisitions_used: int
    acquisitions_remaining: int
    acquisition_limit: int
    acquisition_warning: bool
    il_used: int
    il_total: int
    ip_accumulated: float
    ip_minimum: float
    ip_pace: IPPaceFlag
    as_of: datetime

    class Config:
        frozen = True
```

#### P0-3: FreshnessMetadata

```python
class FreshnessMetadata(BaseModel):
    """Per-response freshness annotation. Every API response must include this."""
    primary_source: str
    fetched_at: Optional[datetime]
    computed_at: datetime
    staleness_threshold_minutes: int
    is_stale: bool

    class Config:
        frozen = True
```

#### P0-4: CategoryStats (with validator)

```python
from pydantic import field_validator

class CategoryStats(BaseModel):
    """Stats for a single time window across all scoring categories."""
    values: Dict[str, Optional[float]]

    @field_validator('values')
    @classmethod
    def validate_category_keys(cls, v: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        actual = set(v.keys())
        missing = SCORING_CATEGORY_CODES - actual
        if missing:
            raise ValueError(f"Missing scoring categories: {missing}")
        extra = actual - SCORING_CATEGORY_CODES
        if extra:
            raise ValueError(f"Unexpected category keys: {extra}")
        return v

    class Config:
        frozen = True
```

**CRITICAL:** The validator references `SCORING_CATEGORY_CODES` which is derived from the loaded contract at import time. This means category validation is always consistent with the JSON — if you add a category to the JSON, the validator adapts automatically. No hardcoded frozensets.

#### P0-5: MatchupScoreboardRow + MatchupScoreboardResponse

```python
class MatchupScoreboardRow(BaseModel):
    """One row per scoring category on the matchup scoreboard. Fields map to MS-1 through MS-12."""
    category: str                        # MS-1: canonical code (e.g. "HR_B", not "HR")
    category_label: str                  # MS-1: short_label from contract (e.g. "HR")
    is_lower_better: bool                # From contract direction field
    is_batting: bool                     # True if batting, False if pitching
    my_current: float                    # MS-2
    opp_current: float                   # MS-3
    current_margin: float                # MS-4: signed (positive = winning, respects direction)
    # --- Phase 2/3 fields: Optional until ROW projections + Monte Carlo are wired ---
    my_projected_final: Optional[float]  # MS-5
    opp_projected_final: Optional[float] # MS-6
    projected_margin: Optional[float]    # MS-7
    status: Optional[CategoryStatusTag]  # MS-8
    flip_probability: Optional[float]    # MS-9
    delta_to_flip: Optional[str]         # MS-10
    games_remaining: Optional[int]       # MS-11
    ip_context: Optional[str]            # MS-12

    class Config:
        frozen = True

class MatchupScoreboardResponse(BaseModel):
    """Full matchup scoreboard. Returned by GET /api/fantasy/scoreboard."""
    week: int
    opponent_name: str
    categories_won: int                         # MS-13
    categories_lost: int                        # MS-13
    categories_tied: int                        # MS-13
    projected_won: Optional[int]                # MS-14
    projected_lost: Optional[int]               # MS-14
    projected_tied: Optional[int]               # MS-14
    overall_win_probability: Optional[float]    # MS-15
    rows: List[MatchupScoreboardRow]            # 18 rows, one per scoring category
    budget: ConstraintBudget
    freshness: FreshnessMetadata

    class Config:
        frozen = True
```

#### P0-6: CanonicalPlayerRow + PlayerGameContext

```python
class PlayerGameContext(BaseModel):
    """Today's game context for a player."""
    opponent: str
    home_away: str                               # "home" or "away"
    game_time: Optional[datetime]
    # Pitcher-specific
    projected_k: Optional[float]                 # PR-7
    projected_era_impact: Optional[float]        # PR-8
    # Hitter-specific
    opposing_sp_name: Optional[str]              # PR-11
    opposing_sp_handedness: Optional[str]        # PR-11: "L" or "R"
    projected_impact: Optional[float]            # PR-12

    class Config:
        frozen = True

class CanonicalPlayerRow(BaseModel):
    """Universal player representation. Fields map to PR-1 through PR-22."""
    # Identity
    player_name: str                             # PR-1
    team: str                                    # PR-2
    eligible_positions: List[str]                # PR-3
    # Status
    status: str                                  # PR-4: playing|not_playing|probable|IL|minors
    game_context: Optional[PlayerGameContext]     # PR-5 through PR-12
    # Stats by window (keyed by canonical codes via CategoryStats validator)
    season_stats: Optional[CategoryStats]        # PR-13
    rolling_7d: Optional[CategoryStats]          # PR-14
    rolling_15d: Optional[CategoryStats]         # PR-15
    rolling_30d: Optional[CategoryStats]         # PR-16
    ros_projection: Optional[CategoryStats]      # PR-17
    row_projection: Optional[CategoryStats]      # PR-18 (Phase 2 deliverable)
    # Metadata
    ownership_pct: Optional[float]               # PR-19
    injury_status: Optional[str]                 # PR-20
    injury_return_timeline: Optional[str]        # PR-21
    freshness: FreshnessMetadata                 # PR-22
    # Internal IDs (not displayed)
    yahoo_player_key: Optional[str] = None
    bdl_player_id: Optional[int] = None
    mlbam_id: Optional[int] = None

    class Config:
        frozen = True
```

---

## 6. TESTS

### `tests/test_stat_contract_schema.py`

1. **Contract loads successfully** — `load_contract()` returns a `FantasyStatContract` instance
2. **Exactly 18 scoring stats** — `len(stats where is_scoring_category) == 18`
3. **Batting/pitching split** — 9 batting + 9 pitching, no overlap, complete coverage
4. **Lower-is-better correct** — exactly `ERA`, `WHIP`, `L`, `K_B`, `HR_P`
5. **Weekly rules validate** — `pitcher_ip_minimum_outs == round(pitcher_ip_minimum * 3)`
6. **Yahoo ID index complete** — every ID maps to a stat that exists in stats or supporting_stats
7. **Matchup display order** — all 18 scoring categories present, plus display-only stats
8. **Cross-validation** — scoring_categories.batting codes match stats with `scoring_role == "batting"`
9. **Schema rejects bad data** — missing required field raises `ValidationError`
10. **Schema rejects extra fields** — unknown field raises `ValidationError` (extra="forbid")
11. **Convenience constants** — `SCORING_CATEGORY_CODES` has 18 elements, `BATTING_CODES` has 9, `PITCHING_CODES` has 9, `LOWER_IS_BETTER` has 5

### `tests/test_ui_contracts.py`

1. **CategoryStatusTag** — has exactly 5 values; all values are lowercase strings
2. **ConstraintBudget** — create with sample data; frozen (mutation raises)
3. **FreshnessMetadata** — create with sample data; verify frozen
4. **CategoryStats validator** — 18 canonical-code keys succeeds; missing key raises; extra key raises
5. **CategoryStats uses contract codes** — validate that the validator references `SCORING_CATEGORY_CODES` (not hardcoded)
6. **MatchupScoreboardRow** — instantiate with required fields; frozen
7. **MatchupScoreboardResponse** — instantiate with 18 rows + budget + freshness
8. **CanonicalPlayerRow** — create with full data; verify frozen
9. **PlayerGameContext** — create with sample data; verify frozen
10. **All contracts frozen** — verify mutation raises error for every contract class

---

## 7. GATE 0 VERIFICATION

After implementation, run:

```bash
venv/Scripts/python -m py_compile backend/stat_contract/schema.py
venv/Scripts/python -m py_compile backend/stat_contract/registry.py
venv/Scripts/python -m py_compile backend/stat_contract/builder.py
venv/Scripts/python -m py_compile backend/stat_contract/loader.py
venv/Scripts/python -m py_compile backend/stat_contract/__init__.py
venv/Scripts/python -m py_compile backend/contracts.py
venv/Scripts/python -m pytest tests/test_stat_contract_schema.py tests/test_ui_contracts.py -q --tb=short
```

All must pass with zero failures. Gate 0 checklist:

- [ ] `fantasy_stat_contract.json` loads and validates against the Pydantic schema
- [ ] Schema catches bad data (missing fields, extra fields, wrong types)
- [ ] `SCORING_CATEGORY_CODES` has exactly 18 elements, derived from loaded contract
- [ ] `LOWER_IS_BETTER` has exactly 5 elements: `ERA`, `WHIP`, `L`, `K_B`, `HR_P`
- [ ] `YAHOO_ID_INDEX` maps all Yahoo stat IDs to canonical codes
- [ ] `CategoryStats` validator rejects keys not in the contract
- [ ] `CategoryStatusTag` enum has all 5 values with threshold definitions in docstring
- [ ] `ConstraintBudget` has all fields from GH-6 through GH-14 per the audit
- [ ] `MatchupScoreboardRow` has all fields from MS-1 through MS-12 per the audit
- [ ] `MatchupScoreboardResponse` has header fields MS-13 through MS-16
- [ ] `CanonicalPlayerRow` has all fields from PR-1 through PR-22 per the audit
- [ ] Every `py_compile` passes
- [ ] All test files pass
- [ ] No UI contract allows `None` for fields the UI spec requires unconditionally

---

## 8. WHAT YOU MUST NOT DO

1. **Do not modify existing contracts** — `UncertaintyRange`, `LineupOptimizationRequest`, `PlayerValuationReport`, `ExecutionDecision` are untouched
2. **Do not create API endpoints, DB migrations, or data pipelines** — type definitions and a loader only
3. **Do not hardcode category keys as frozensets** — derive ALL category constants from the loaded contract JSON
4. **Do not modify `backend/utils/fantasy_stat_contract.py` or `backend/utils/fantasy_stat_contract.json`** — the old v1 loader stays until consumers are migrated in a later phase
5. **Do not import from models.py, db.py, or services** — the stat_contract package and UI contracts must be self-contained
6. **Do not touch HANDOFF.md** — update it only after Gate 0 passes
7. **Do not abbreviate the registry** — copy every stat definition from the example with every field, note, and external ID intact
8. **Do not use Pydantic v1 style in the stat_contract package** — use `model_config = ConfigDict(...)` (Pydantic v2 native). The existing contracts in `backend/contracts.py` use v1 `class Config:` style — that's fine, leave them as-is and use the same style for the new UI contracts added there for consistency.

---

## 9. AFTER GATE 0 PASSES

Once all tests pass and py_compile succeeds:

1. Update HANDOFF.md:
   - Change Phase 0 status from "NEXT" to "COMPLETE"
   - Change Phase 1 status from "Blocked by Phase 0" to "NEXT"
   - Note: "v1 consumer migration (backend/utils/fantasy_stat_contract.py → backend/stat_contract) deferred to Phase 1"
   - Update "Last Updated" timestamp

2. Do NOT start Phase 1 in this session. Gate 0 completion and HANDOFF.md update is the deliverable.

**Phase 1 preview** (for context only — do not implement):
- Migrate `backend/utils/fantasy_stat_contract.py` consumers to `backend/stat_contract`
- Wire `backend/main.py` and `backend/routers/fantasy.py` imports to new package
- Close 7 L2 data gaps (acquisition counter, IP extractor, etc.)
- Add `scripts/generate_stat_contract.py` CLI (adapted from `Examples/generate_stat_contract.py`)

---

*End of prompt. This prompt is self-contained. No prior conversation context is assumed.*
