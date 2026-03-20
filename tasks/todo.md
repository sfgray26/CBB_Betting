# CBB Edge — Platform Expansion Roadmap
*Updated: 2026-03-20 | Architect: Claude Sonnet 4.6*

---

## Frontend Migration Status

| Phase | Status | Pages |
|-------|--------|-------|
| Phase 0 — Foundation | ✅ DONE | scaffold, auth, layout, design system |
| Phase 1 — Core Analytics | ✅ DONE | /performance, /clv, /bet-history, /calibration, /alerts |
| Phase 2 — Trading | ✅ DONE | /today, /live-slate, /odds-monitor |
| Phase 3 — Tournament | ✅ DONE | /bracket |
| Phase 4 — Mobile & PWA | ✅ DONE | viewport meta, manifest, mobile drawer, responsive grids |
| Phase 5 — Polish & Decommission | ⏳ Future | error boundaries, loading states, retire Streamlit |

### Phase 4 Completed (Mar 20)
- [x] Viewport meta + `appleWebApp` in `app/layout.tsx`
- [x] PWA manifest at `frontend/public/manifest.json`
- [x] PWA icons (192, 512, 512-maskable) in `frontend/public/icons/`
- [x] Mobile sidebar drawer in `sidebar.tsx` (slide-in with `isOpen`/`onClose`)
- [x] Mobile overlay in `(dashboard)/layout.tsx`
- [x] Hamburger menu in `header.tsx` (`md:hidden`)
- [x] Touch targets `min-h-[44px]` on sidebar nav links
- [x] DataTable with `overflow-x-auto` wrapper
- [x] Responsive grids: `grid-cols-1 sm:grid-cols-2` fallbacks on all KPI rows
  - calibration/page.tsx, clv/page.tsx, bracket/page.tsx
  - performance/page.tsx, today/page.tsx, odds-monitor/page.tsx

---

## Betting Model — GUARDIAN FREEZE 🔒

**DO NOT TOUCH until Apr 7, 2026:**
- `backend/betting_model.py`
- `backend/services/analysis.py`
- Any CBB model services

Post-Apr 7 planned (see HANDOFF.md Section 5):
- [ ] V9.2 recalibration (spec: `reports/K12_RECALIBRATION_SPEC_V92.md`)
  - `sd_mult` 1.0 → 0.80, `ha` 2.419 → 2.85, `SNR_KELLY_FLOOR` 0.50 → 0.75
- [ ] Wire Haslametrics as 3rd rating source (`backend/services/haslametrics.py` already built)
- [ ] Add `pricing_engine` field to Prediction model (K-14)
- [ ] Bump `model_version` to 'v9.2', confirm BET rate 3% → 8-12%

---

## Phase 0 — Foundation Hardening (Sprint 1)

- [ ] Set `RAILWAY_TOKEN` and `NEXT_PUBLIC_API_URL` in GitHub Secrets
- [ ] Confirm Railway DB connection string in production env
- [ ] Add `FANTASY_BASEBALL_API_KEY` to `.env.example`
- [ ] Tag current release as `v0.8.0-cbb-stable`

---

## Phase 1 — Fantasy Baseball Integration (Sprint 2–3)

### 1.1 Database Schema
```sql
-- New tables (add to backend/models.py)
CREATE TABLE fantasy_players (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    team VARCHAR(50),
    position VARCHAR(20),
    injury_status VARCHAR(20) DEFAULT 'active',
    adp FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE fantasy_projections (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) REFERENCES fantasy_players(player_id),
    projection_date DATE NOT NULL,
    source VARCHAR(30) NOT NULL,
    projected_hr FLOAT, projected_avg FLOAT, projected_ops FLOAT,
    projected_sb FLOAT, projected_era FLOAT, projected_whip FLOAT,
    projected_k9 FLOAT, projected_sv FLOAT, fantasy_points_est FLOAT,
    UNIQUE(player_id, projection_date, source)
);

CREATE TABLE fantasy_lineups (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    lineup_date DATE NOT NULL,
    platform VARCHAR(30),
    positions JSONB,
    projected_points FLOAT,
    actual_points FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 1.2 Tasks
- [ ] DB Schema: Add `FantasyPlayer`, `FantasyProjection`, `FantasyLineup` models to `backend/models.py`
- [ ] Update `scripts/init_db.py` for new tables
- [ ] Add Pydantic schemas to `backend/schemas.py`
- [ ] Service: Create `backend/services/fantasy.py`
- [ ] API Routes: Add 6 fantasy endpoints to `backend/main.py`
- [ ] Frontend: Create `/fantasy/draft`, `/fantasy/lineup`, `/fantasy/standings` pages

---

## Phase 2 — Advanced Betting Markets (Sprint 4–5)

- [ ] Moneyline model: Create `backend/services/moneyline.py`
- [ ] Player props: Create `backend/services/props.py` with Odds API integration
- [ ] Frontend: Create `/props` page

---

## Phase 3 — Discord Bot (Sprint 6)

- [ ] Create `backend/services/discord_bot.py` with `/picks`, `/bankroll`, `/performance`
- [ ] Create `backend/discord_bot_runner.py` entry point
- [ ] Add `discord.py` and `httpx` to `requirements.txt`

---

## Phase 4 — Admin Risk Dashboard (Sprint 7)

- [ ] Create `/admin/risk` page (portfolio exposure, drawdown gauge)
- [ ] Create `/admin/odds-monitor` page (line movements, quota status)
- [ ] Create `/admin/model` page (read-only V9.1 parameters)

---

## Phase 5 — OpenClaw Autonomous Ops (Sprint 8+)

- [ ] Create `backend/services/openclaw.py` orchestrator
- [ ] Implement multi-agent pipeline with `asyncio.gather`
- [ ] Add integration test `tests/test_openclaw.py`

---

## Known Issues / Blockers

| Issue | Owner | Status |
|-------|-------|--------|
| V9.1 calibration mismatch (SNR stacking) | Claude Code + Kimi | Deferred to Apr 7 |
| EvanMiya down → 2-source mode | Gemini (G-R7) | In progress |
| RAILWAY_TOKEN / NEXT_PUBLIC_API_URL not in GH Secrets | DevOps | Phase 0 |

---

## Done Archive

- Phase 4 Mobile & PWA — Mar 20
- Phase 2+3 Frontend (trading + bracket pages) — Mar 19
- Phase 1 all 5 analytics pages fixed — Mar 18
- Frontend scaffold (Phase 0) — Mar 18
- Railway CORS fix — Mar 18
- BDL odds 400 fix — Mar 18
- Monte Carlo bracket simulator — Mar 16
- Discord morning brief + EOD results — Mar 16
- Team mapping hardening (29 St variants, 78 tests) — Mar 16
- Duplicate bet cleanup endpoint — Mar 16
- V9.1 model (fatigue, sharp money, conf HCA, recency) — Mar 11–12
