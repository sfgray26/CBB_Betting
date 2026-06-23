# UAT Phase 1: Critical Data Fixes — Two-Start Pitcher Identifier

**Date:** 2026-06-12
**Priority:** P0 (High impact, light effort)
**Baseball IQ Impact:** 6.5/10 → 7.0/10

---

## Problem Statement

Weekly H2H leagues depend on two-start pitchers for category dominance. The app does NOT identify which pitchers have 2 starts in the upcoming week.

**UAT Finding:** "In a weekly league (Week 12), a two-start pitcher is critical context. The Waiver Wire shows Jakob Junis, TEX but no indication whether he has 1 or 2 starts this week, which changes his value by 50%+."

**Impact:**
- Managers miss opportunities to add high-value 2-start pitchers
- Pitcher Match Scores underrepresent true value (volume multiplier not applied)
- Streaming decisions suboptimal without start count context

---

## Solution: MLB Schedule Integration + Start Count Badge

### Design

1. **Query MLB schedule API:**
   - Use existing MLB GameDay API integration
   - Query next 7 days of starts per pitcher
   - Cache results in `player_schedule` table

2. **Display start count badge:**
   - `"🗓️ 2 Starts"` for pitchers with 2 starts
   - `"🗓️ 1 Start"` for pitchers with 1 start
   - No badge for position players

3. **Adjust projections:**
   - 2-start pitchers: multiply projection by 1.5x (volume bonus)
   - Display adjusted projection: `"2.4 pts (1.6 pts per start x 2)"`

4. **Add matchup strength:**
   - Query opponent team ERA for each start
   - Display: `"vs NYY (3.12 ERA) • vs BOS (4.25 ERA)"`
   - Color-code: green for weak pitching, red for strong pitching

---

## Implementation Tasks

### Task 3.1: Create PlayerSchedule Model

**File:** `backend/models.py`

Add new table:
```python
class PlayerSchedule(Base):
    """
    P20 Player schedule tracking — weekly start counts for pitchers.

    Stores next 7 days of starts per player, updated daily.

    Natural key: (bdl_player_id, as_of_date).
    """
    __tablename__ = "player_schedule"

    id = Column(BigInteger, primary_key=True)
    bdl_player_id = Column(Integer, nullable=False)
    as_of_date = Column(Date, nullable=False)
    player_type = Column(String(10), nullable=False)

    # Start counts for next 7 days
    starts_next_7d = Column(Integer, nullable=False, default=0)

    # Start details (JSON array of game objects)
    start_details = Column(JSON, nullable=True)

    computed_at = Column(DateTime(timezone=True), nullable=False, default=_now_et)

    __table_args__ = (
        UniqueConstraint("bdl_player_id", "as_of_date", name="_ps_player_date_uc"),
        Index("idx_ps_starts", "starts_next_7d"),
    )
```

### Task 3.2: Query MLB GameDay API

**File:** `backend/services/schedule_fetcher.py`

Create new function:
```python
async def fetch_pitcher_starts_next_7d(
    db: Session,
    as_of_date: date,
    mlb_client: MLBGameDayClient,
) -> List[PlayerSchedule]:
    """
    Query MLB GameDay API for pitcher starts in next 7 days.

    Returns list of PlayerSchedule objects with start counts and details.
    """
    # Query MLB schedule for next 7 days
    # Parse starting pitchers for each game
    # Group by bdl_player_id
    # Return PlayerSchedule objects
```

### Task 3.3: Update Daily Ingestion Pipeline

**File:** `backend/services/daily_ingestion.py`

Add new cron job (lock 100_038, 5:45 AM ET):
```python
async def _compute_player_schedule(self) -> dict:
    """
    Daily player schedule computation (lock 100_038, 5:45 AM ET).

    Runs after _compute_player_momentum (5 AM) so momentum is current.

    Algorithm:
      1. Query MLB GameDay API for next 7 days
      2. Parse starting pitchers
      3. Build PlayerSchedule objects
      4. Upsert to player_schedule table
    """
```

### Task 3.4: Update Waiver Response Schema

**File:** `backend/contracts.py` (WaiverPlayerOut)

Add schedule fields:
```python
starts_next_7d: Optional[int] = None
start_details: Optional[List[dict]] = None
adjusted_projection: Optional[float] = None  # Projection x volume multiplier
```

### Task 3.5: Update Waiver Wire Endpoint

**File:** `backend/routers/fantasy.py` (waiver_recommendations)

Add schedule lookup:
```python
# For each pitcher, query player_schedule
if player.player_type == "pitcher":
    schedule = db.query(PlayerSchedule).filter(
        PlayerSchedule.bdl_player_id == player.bdl_player_id,
        PlayerSchedule.as_of_date == today,
    ).first()

    if schedule and schedule.starts_next_7d > 1:
        # Adjust projection by volume multiplier
        adjusted_projection = player.projection * schedule.starts_next_7d * 1.5
```

### Task 3.6: Update Frontend Display

**File:** `frontend/components/waiver/player-row.tsx`

Add start count badge:
```tsx
{player.player_type === "pitcher" && player.starts_next_7d > 0 && (
  <Tooltip content={player.starts_next_7d > 1 ? "High volume pitcher" : "Single start"}>
    <Badge variant={player.starts_next_7d > 1 ? "success" : "default"}>
      🗓️ {player.starts_next_7d} Start{player.starts_next_7d > 1 ? "s" : ""}
    </Badge>
  </Tooltip>
)}

{player.adjusted_projection != null && player.adjusted_projection !== player.projection && (
  <div className="text-xs text-text-tertiary">
    {player.adjusted_projection.toFixed(1)} pts
    <span className="text-text-muted">
      (1.6 x {player.starts_next_7d} starts)
    </span>
  </div>
)}
```

---

## Testing Strategy

### Unit Tests
1. Test schedule parsing: verify MLB GameDay API response parsed correctly
2. Test start counting: verify 2-start pitchers counted correctly
3. Test projection adjustment: verify 1.5x multiplier applied

### Integration Tests
1. Run schedule fetch: verify PlayerSchedule objects created
2. Test waiver endpoint: verify schedule fields returned
3. Test volume multiplier: verify adjusted_projection > projection for 2-start pitchers

### Regression Tests
1. Ensure position players unaffected (no start badge shown)
2. Verify existing Match Score works (backward compat)

---

## Rollout Plan

1. **Create PlayerSchedule table** (migration)
2. **Implement schedule_fetcher.py** with MLB GameDay API query
3. **Add daily cron job** (lock 100_038, 5:45 AM ET)
4. **Deploy backend** with schedule fields
5. **Deploy frontend** with start badges
6. **Monitor** for API failures (log MLB GameDay errors)

---

## Success Metrics

- **Schedule coverage:** >95% of pitchers have start counts for next 7 days
- **2-start detection:** 100% of 2-start pitchers flagged
- **Volume multiplier:** Adjusted projections 50%+ higher for 2-start pitchers
- **User adoption:** Survey shows >80% use start badges in decisions

---

## Dependencies

- **MLB GameDay API** (already integrated in daily_ingestion.py)
- **player_id_mapping** (maps MLBAM ID to BDL ID)

---

## Time Estimate

- Create PlayerSchedule model + migration: 1 hour
- Implement schedule_fetcher.py: 4 hours
- Add daily cron job: 2 hours
- Update waiver endpoint: 2 hours
- Frontend display: 2 hours
- Testing: 2 hours

**Total: 13 hours (2 days)**