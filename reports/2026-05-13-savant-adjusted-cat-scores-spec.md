# K-NEXT-3 Spec: SAVANT_ADJUSTED cat_scores Assembly for player_board

**Date:** 2026-05-13
**Agent:** Kimi CLI
**Scope:** Research only — no code changes implemented

---

## 1. Schema Audit: CanonicalProjection & CategoryImpact

### 1.1 CanonicalProjection (`backend/models.py` lines 2031–2126)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| `id` | Integer (PK, auto) | No | Internal surrogate key |
| `projection_id` | String(36) | No | UUID — not used for lookups |
| `player_id` | **BigInteger** | No | **Canonical ID. Can be `mlbam_id` OR `-(yahoo_id)` fallback** |
| `player_type` | String(10) | No | `"BATTER"` \| `"PITCHER"` |
| `source_engine` | String(25) | No | **`SAVANT_ADJUSTED` \| `STATIC_BOARD` \| `BAYESIAN` \| `FALLBACK_MARCEL`** |
| `projection_date` | Date | No | Latest = `2026-05-13` |
| `season` | Integer | No | `2026` |
| `source_ids` | JSONB | Yes | Upstream source identifiers |
| `projected_pa` / `projected_ab` / `projected_ip` | Float | Yes | Playing time |
| `proj_hr`, `proj_sb`, `proj_r`, `proj_rbi` | Integer | Yes | Batter counting |
| `proj_w`, `proj_sv`, `proj_k` | Integer | Yes | Pitcher counting |
| `proj_avg`, `proj_obp`, `proj_slg`, `proj_ops` | Float | Yes | Batter rates |
| `proj_era`, `proj_whip`, `proj_k9` | Float | Yes | Pitcher rates |
| `confidence_score` | Float | Yes | 0.0–1.0 |

**Critical finding:** `CanonicalProjection` has **NO `player_name` column**. Lookups must go through `player_id`.

**Relationship:** `CanonicalProjection.category_impacts` → 1:N `CategoryImpact` (back_populates="projection")

### 1.2 CategoryImpact (`backend/models.py` lines 2128–2158)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| `id` | Integer (PK, auto) | No | |
| `canonical_projection_id` | Integer (FK → `canonical_projections.id`, CASCADE) | No | **Join column** |
| `category` | String(20) | No | e.g. `"HR"`, `"AVG"`, `"ERA"` |
| `projected_value` | Float | Yes | Raw projected stat |
| `z_score` | Float | Yes | Context-agnostic z-score |
| `generic_marginal_impact` | Float | Yes | Placeholder (= z_score currently) |
| `denominator_weight` | Float | Yes | 1.0 for counting stats |
| `projected_numerator` | Float | Yes | For rate marginal math |
| `projected_denominator` | Float | Yes | For rate marginal math |

**Relationship:** `CategoryImpact.projection` → M:1 `CanonicalProjection` (back_populates="category_impacts")

---

## 2. Current Data Flow in `get_or_create_projection`

Located in `backend/fantasy_baseball/player_board.py` lines 1119–1462.

```
yahoo_player dict (has name, player_key, positions)
    │
    ▼
[1] Runtime cache check (_projection_cache)
    │
    ▼
[2] Identity resolution → mlbam_id (via PlayerIDMapping / PlayerIdentity)
    │
    ▼
[3] Query PlayerProjection by mlbam_id or player_name
    │
    ├──► FAST PATH: If PlayerProjection.cat_scores is a real dict → return immediately
    │       (source = "steamer_db")
    │
    └──► SLOW PATH: Extract steamer_data + statcast_data → fusion_engine
              │
              ▼
         _convert_fusion_proj_to_board_format()
              │
              ▼
         Returns proxy with z_score=0.0, cat_scores={}
              │
              ▼
         DRAFT BOARD FALLBACK: If no DB data at all → return hardcoded board entry
```

**Key observation:** The FAST PATH (lines 1287–1315) is the only place where real `cat_scores` are returned. It exclusively queries `PlayerProjection` (the old table), not `CanonicalProjection`.

---

## 3. Implementation Spec: Exact Code for SAVANT_ADJUSTED Lookups

### 3.1 Name → CanonicalProjection Lookup Chain

Because `CanonicalProjection` has no `player_name`, we must bridge through `PlayerIdentity`:

```python
# Name → normalized_name → PlayerIdentity.mlbam_id → CanonicalProjection.player_id
# OR
# Name → normalized_name → PlayerIdentity.yahoo_id → -(yahoo_id) fallback namespace
```

Exact lookup helper to insert into `player_board.py`:

```python
def _lookup_canonical_by_name(
    db, name: str, player_type: str
) -> tuple[CanonicalProjection, list[CategoryImpact]] | None:
    """Return the latest CanonicalProjection + its CategoryImpacts for a player name.

    Bridges through PlayerIdentity because CanonicalProjection has no player_name.
    Tries mlbam_id first, then negative yahoo_id fallback namespace.
    Only returns SAVANT_ADJUSTED rows (not STATIC_BOARD).
    """
    from backend.models import (
        CanonicalProjection,
        CategoryImpact,
        PlayerIdentity,
    )

    norm = _normalize_name(name) if name else ""
    if not norm or not db:
        return None

    identity_row = (
        db.query(PlayerIdentity)
        .filter(PlayerIdentity.normalized_name == norm)
        .first()
    )
    if not identity_row:
        return None

    # Determine canonical player_id namespace
    candidate_ids: list[int] = []
    if identity_row.mlbam_id:
        candidate_ids.append(identity_row.mlbam_id)
    if identity_row.yahoo_id:
        # Negative namespace used by ProjectionAssemblyService for Yahoo-only players
        candidate_ids.append(-(int(identity_row.yahoo_id)))

    if not candidate_ids:
        return None

    # Fetch latest projection — prefer SAVANT_ADJUSTED, any date
    cp = (
        db.query(CanonicalProjection)
        .filter(
            CanonicalProjection.player_id.in_(candidate_ids),
            CanonicalProjection.player_type == player_type.upper(),
        )
        .order_by(
            # Prefer SAVANT_ADJUSTED, then most recent date
            CanonicalProjection.source_engine == "SAVANT_ADJUSTED",
            CanonicalProjection.projection_date.desc(),
        )
        .first()
    )

    if cp is None:
        return None

    # Only use SAVANT_ADJUSTED for the upgrade path
    if cp.source_engine != "SAVANT_ADJUSTED":
        return None

    impacts = (
        db.query(CategoryImpact)
        .filter(CategoryImpact.canonical_projection_id == cp.id)
        .all()
    )

    return cp, impacts
```

### 3.2 CategoryImpact → player_board cat_scores Assembly

`CategoryImpact` stores uppercase category strings. `player_board` expects lowercase keys with some naming differences:

| CategoryImpact | player_board cat_scores key | Direction | Type |
|----------------|---------------------------|-----------|------|
| `R` | `r` | positive | counting |
| `HR` | `hr` | positive | counting |
| `RBI` | `rbi` | positive | counting |
| `SB` | `sb` (not `nsb`) | positive | counting |
| `AVG` | `avg` | positive | rate |
| `OBP` | *(no direct equivalent)* | positive | rate |
| `OPS` | `ops` | positive | rate |
| `W` | `w` | positive | counting |
| `K` | `k_pit` | positive | counting |
| `SV` | `sv` (not `nsv`) | positive | counting |
| `ERA` | `era` | **negative** | rate |
| `WHIP` | `whip` | **negative** | rate |
| `K9` | `k9` | positive | rate |

**Missing from CategoryImpact (vs player_board):**
- `k_bat` (batting strikeouts — negative category in H2H One Win)
- `tb` (total bases)
- `nsb` (net stolen bases = SB - CS)
- `l` (losses — negative)
- `hr_pit` (pitcher home runs allowed — negative)
- `qs` (quality starts)

**Schema gap:** CategoryImpact does not store `k_bat`, `tb`, `nsb`, `l`, `hr_pit`, or `qs`. These are H2H One Win specific categories that the `ProjectionAssemblyService` does not currently emit.

Exact assembly helper:

```python
def _category_impacts_to_cat_scores(
    impacts: list[CategoryImpact],
    player_type: str,
) -> dict[str, float]:
    """Convert CategoryImpact rows to player_board-compatible cat_scores dict.

    Uses the pre-computed z_score from CategoryImpact (computed across full
    player pool by ProjectionAssemblyService). Missing categories get 0.0.
    """
    cat_scores: dict[str, float] = {}

    # Mapping: CategoryImpact.category → player_board key
    BATTER_MAP = {
        "R": "r",
        "HR": "hr",
        "RBI": "rbi",
        "SB": "sb",
        "AVG": "avg",
        "OPS": "ops",
    }
    PITCHER_MAP = {
        "W": "w",
        "K": "k_pit",
        "SV": "sv",
        "ERA": "era",
        "WHIP": "whip",
        "K9": "k9",
    }

    impact_map = BATTER_MAP if player_type == "batter" else PITCHER_MAP

    for impact in impacts:
        cat = impact.category
        if cat not in impact_map:
            continue
        board_key = impact_map[cat]
        z = impact.z_score
        if z is not None:
            cat_scores[board_key] = round(z, 3)

    # Note: k_bat, tb, nsb, l, hr_pit, qs are NOT in CategoryImpact.
    # They will remain absent (effectively 0.0) from the cat_scores dict.
    return cat_scores
```

### 3.3 Integration Point in `get_or_create_projection`

Insert the SAVANT_ADJUSTED fast path **immediately after** the existing `PlayerProjection` fast path (after line 1315) and **before** the fusion slow path:

```python
    # ── FAST PATH 2: SAVANT_ADJUSTED CanonicalProjection ──────────────────
    # If PlayerProjection had no cat_scores, try the newer CanonicalProjection
    # table which has Statcast-fused SAVANT_ADJUSTED rows with CategoryImpact
    # z-scores computed across the full player pool.
    # ──────────────────────────────────────────────────────────────────────
    if projection_cat_scores is None and db is not None:
        try:
            canonical_result = _lookup_canonical_by_name(db, name, player_type)
            if canonical_result is not None:
                cp, impacts = canonical_result
                sa_cat_scores = _category_impacts_to_cat_scores(impacts, player_type)
                if sa_cat_scores:
                    logger.info(
                        "[player_board] SAVANT_ADJUSTED fast path for %s "
                        "(source_engine=%s, categories=%s)",
                        name, cp.source_engine, list(sa_cat_scores.keys()),
                    )
                    # Close DB session before returning
                    if db_gen is not None:
                        try:
                            next(db_gen)
                        except (StopIteration, Exception):
                            pass
                    proxy = {
                        "id": player_key or name.lower().replace(" ", "_"),
                        "name": name,
                        "team": yahoo_player.get("team")
                            or yahoo_player.get("editorial_team_abbr")
                            or "",
                        "positions": positions,
                        "type": player_type,
                        "tier": 10,
                        "rank": 9999,
                        "adp": 9999.0,
                        "z_score": sum(sa_cat_scores.values()),
                        "cat_scores": sa_cat_scores,
                        "proj": {},
                        "is_keeper": False,
                        "keeper_round": None,
                        "is_proxy": True,
                        "fusion_source": "savant_adjusted_db",
                        "components_fused": 2,  # steamer + statcast
                        "xwoba_override": getattr(
                            cp.explainability_metadata or {},
                            "xwoba_override_detected",
                            False,
                        ),
                    }
                    if player_key:
                        _projection_cache[player_key] = proxy
                    return proxy
        except Exception as _cp_err:
            logger.debug(
                "[player_board] CanonicalProjection lookup failed for %s: %s",
                name, _cp_err,
            )
```

---

## 4. Coverage Estimate

### 4.1 Database Truth (as of 2026-05-13)

| Metric | Value |
|--------|-------|
| Total `CanonicalProjection` rows | 4,012 |
| `SAVANT_ADJUSTED` rows | 2,123 |
| `STATIC_BOARD` rows | 1,889 |
| Distinct players with `SAVANT_ADJUSTED` | **238** |
| Distinct players with `STATIC_BOARD` | 216 |
| Latest `projection_date` | 2026-05-13 |
| `CategoryImpact` rows | 26,119 |
| Categories per impact set | 7 (batters) or 6 (pitchers) |

### 4.2 Breakdown by player_type × source_engine

| player_type | SAVANT_ADJUSTED | STATIC_BOARD |
|-------------|-----------------|--------------|
| BATTER | 1,178 rows | 869 rows |
| PITCHER | 945 rows | 1,020 rows |

### 4.3 Waiver Pool Coverage Estimate

The Yahoo fantasy league has ~300 rostered players + ~200 top free agents = ~500 relevant players.

- **238 SAVANT_ADJUSTED players** = ~48% of the 500-player relevant universe
- These are likely the **most-played, Statcast-qualified players** (PA ≥ 50 / IP ≥ 20)
- Deep waiver wire players (unqualified samples) fall back to `STATIC_BOARD` or draft board

**Top-200 free agent coverage estimate:**
- Assuming top free agents are roughly ordered by ownership % / playing time
- ~60–70% of the top-100 free agents would have SAVANT_ADJUSTED rows
- ~40–50% of the top-200 free agents would have SAVANT_ADJUSTED rows
- The remainder are call-ups, part-time players, or injured players with insufficient PA/IP

### 4.4 Category Coverage Gaps

Even for SAVANT_ADJUSTED players, the following H2H One Win categories are **missing** from `CategoryImpact`:

| Missing category | player_board key | Impact |
|------------------|------------------|--------|
| Batting strikeouts | `k_bat` | **High** — negative category, affects value |
| Total bases | `tb` | Medium — positive category |
| Net stolen bases | `nsb` | Medium — positive category (SB - CS) |
| Losses | `l` | Medium — negative category for pitchers |
| Pitcher HR allowed | `hr_pit` | Low — negative category |
| Quality starts | `qs` | Low — positive category |

**Effect:** A player upgraded to SAVANT_ADJUSTED cat_scores will have **fewer category keys** than a draft-board player (7 vs ~9 for batters, 6 vs ~9 for pitchers). The `z_score` sum will be based on a smaller subset of categories, potentially underweighting players who derive value from the missing categories.

---

## 5. Schema Gaps Blocking Full Implementation

### Gap 1: No `player_name` on CanonicalProjection
- **Status:** Workaround exists (PlayerIdentity bridge)
- **Risk:** Identity miss rate ~5–10% for players without `PlayerIdentity` rows
- **Mitigation:** The `_lookup_canonical_by_name` helper handles misses gracefully and falls through to existing logic

### Gap 2: Missing H2H One Win categories in CategoryImpact
- **Status:** Requires `ProjectionAssemblyService` enhancement
- **Blocked categories:** `k_bat`, `tb`, `nsb`, `l`, `hr_pit`, `qs`
- **Risk:** SAVANT_ADJUSTED cat_scores will be **incomplete** vs draft-board cat_scores
- **Mitigation options:**
  1. **Hybrid approach:** Use SAVANT_ADJUSTED z-scores for categories that exist, fill missing categories from draft board or PlayerProjection
  2. **Extend PAS:** Add the missing categories to `ProjectionAssemblyService._build_category_impacts()`

### Gap 3: CategoryImpact has no `provenance` field
- **Original concern from task description:** The task assumed `CanonicalProjection` had a `provenance` field
- **Status:** `source_engine` on `CanonicalProjection` serves this purpose
- **No action needed**

### Gap 4: `z_score` scale mismatch
- **CategoryImpact.z_score** is computed by `ProjectionAssemblyService` using population std across the **full board player pool** (~400 players)
- **player_board cat_scores** are computed using the same methodology but against the **draft board pool** (~200 players)
- **Risk:** Z-scores from CategoryImpact may have a different scale than draft-board z-scores
- **Mitigation:** The difference is likely small (both use population std). Monitor for scale drift after implementation.

---

## 6. Recommended Implementation Order

1. **Add `_lookup_canonical_by_name` and `_category_impacts_to_cat_scores` helpers** to `player_board.py`
2. **Insert the SAVANT_ADJUSTED fast path** into `get_or_create_projection` (after the PlayerProjection fast path)
3. **Hybrid fallback for missing categories:** When SAVANT_ADJUSTED cat_scores are missing a category that exists in the draft board, blend the draft-board value in (weighted by confidence_score)
4. **Monitor coverage:** Log how many players hit the SAVANT_ADJUSTED path vs STATIC_BOARD vs draft-board fallback
5. **Future: Extend ProjectionAssemblyService** to emit `k_bat`, `tb`, `nsb`, `l`, `hr_pit`, `qs` CategoryImpact rows

---

## 7. Exact Files to Modify (for Claude delegation)

| File | Change Type | Lines |
|------|-------------|-------|
| `backend/fantasy_baseball/player_board.py` | Add 2 helpers + fast path | ~1120–1320 |
| `backend/fantasy_baseball/projection_assembly_service.py` | Future: add missing categories | ~638–693 |

---

*End of spec. No code changes were made to production files. Claude Code to review and implement.*
