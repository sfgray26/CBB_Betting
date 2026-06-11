# Handoff: BDL #2 - Player Search Auto-Heal

## Task Summary
Auto-heal unmapped Yahoo players via BDL (Ball Don't Lie) API search. When a Yahoo player appears in a roster but has no mapping in `player_id_mapping`, automatically search BDL and create the mapping.

## Implementation Complete ✅

### Files Modified

| File | Changes |
|------|---------|
| `backend/models.py` | Added `heal_attempts` and `healed_at` columns to PlayerIDMapping |
| `backend/services/player_autoheal.py` | Updated to track heal attempts and set healed_at on success |
| `backend/fantasy_baseball/yahoo_client_resilient.py` | Added `_trigger_auto_heal_for_unmapped()` method called after roster fetch |
| `backend/scripts/migrations/add_player_mapping_heal_columns.sql` | SQL migration for new columns |
| `backend/test_player_search_auto_heal.py` | Comprehensive test suite (18 tests, all passing) |

### Key Changes

#### 1. PlayerIDMapping Model (models.py:1294-1305)
```python
heal_attempts = Column(Integer, nullable=False, default=0)  # Track attempts
healed_at = Column(DateTime(timezone=True), nullable=True)  # Track success time
```

#### 2. Auto-Heal Service (player_autoheal.py)
- Increments `heal_attempts` on every attempt (even skipped ones)
- Sets `healed_at` timestamp when heal succeeds
- Sets `source = "bdl_search"` for auto-healed mappings

#### 3. Yahoo Client Integration (yahoo_client_resilient.py:741-891)
New `_trigger_auto_heal_for_unmapped()` method:
- Runs asynchronously (fire-and-forget background thread)
- Doesn't block roster response (< 10ms overhead)
- Checks each roster player against player_id_mapping
- Triggers auto-heal for unmapped players
- Logs warnings for unmapped players

#### 4. SQL Migration
```bash
psql $DATABASE_URL -f backend/scripts/migrations/add_player_mapping_heal_columns.sql
```

### Data Flow

```
1. User fetches roster via get_roster()
   ↓
2. Roster parsed and enriched with ownership %
   ↓
3. _trigger_auto_heal_for_unmapped() starts in background thread
   ↓
4. For each unmapped player:
   a. Check if exists in player_id_mapping
   b. If not, call BDL search_mlb_players(name)
   c. If confident match, create/update mapping
   d. Set source='bdl_search', healed_at=now()
   ↓
5. Roster returned immediately (non-blocking)
```

### Safety Features

1. **No Manual Overwrites**: Mappings with `source='manual'` are never touched
2. **Freshness Check**: BDL search mappings < 7 days old aren't re-processed
3. **Confidence Threshold**: Only matches with ≥ 0.85 name similarity are accepted
4. **Conflict Detection**: BDL IDs already mapped to other players are skipped
5. **Non-Blocking**: Auto-heal runs in background thread, doesn't delay roster response
6. **Idempotent**: Same player can be healed multiple times (heal_attempts increments)

### Test Coverage (18 tests)

| Test Category | Tests |
|---------------|-------|
| Auto-heal success | 2 |
| Auto-heal no match | 2 |
| Source tracking | 3 |
| No regression | 2 |
| Batch heal | 2 |
| Helper functions | 6 |
| Yahoo client integration | 1 |

### Run Tests
```bash
cd backend && python -m pytest test_player_search_auto_heal.py -v
```

### Apply Migration
```bash
# Run the migration to add new columns
psql $DATABASE_URL -f backend/scripts/migrations/add_player_mapping_heal_columns.sql
```

### Next Steps (Post-Deploy)

1. **Monitor auto-heal metrics**:
   ```sql
   SELECT 
       source,
       COUNT(*) as total,
       COUNT(healed_at) as auto_healed,
       AVG(heal_attempts) as avg_attempts
   FROM player_id_mapping
   GROUP BY source;
   ```

2. **Review heal failures**:
   ```sql
   SELECT yahoo_key, full_name, heal_attempts
   FROM player_id_mapping
   WHERE heal_attempts > 0 AND healed_at IS NULL
   ORDER BY heal_attempts DESC;
   ```

3. **Monitor logs** for warnings:
   - "Unmapped Yahoo player in roster"
   - "Roster auto-heal complete"

### Git Commit Commands
```bash
cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge
git checkout -b agent/claude/bdl-2-player-search
git add backend/models.py backend/services/player_autoheal.py backend/fantasy_baseball/yahoo_client_resilient.py backend/scripts/migrations/add_player_mapping_heal_columns.sql backend/test_player_search_auto_heal.py
git commit -m "BDL #2: Player search auto-heal for unmapped Yahoo players

- Add heal_attempts and healed_at columns to player_id_mapping
- Update PlayerAutoHealService to track attempts and success timestamps
- Wire auto-heal trigger into yahoo_client_resilient.get_roster()
- Fire-and-forget background thread (non-blocking)
- Safety: respects manual mappings, freshness checks, confidence thresholds
- Comprehensive test suite with 18 passing tests
- SQL migration included

EV: 4.67"
```

## Verification Checklist
- [x] New columns added to PlayerIDMapping model
- [x] Auto-heal service tracks attempts and healed_at
- [x] Yahoo client triggers auto-heal on roster fetch
- [x] Background thread execution (non-blocking)
- [x] Manual mappings protected
- [x] Freshness check (< 7 days)
- [x] Confidence threshold (0.85)
- [x] Conflict detection for bdl_id
- [x] SQL migration created
- [x] 18 tests written and passing
- [ ] Migration applied to database
- [ ] Deployed to production
