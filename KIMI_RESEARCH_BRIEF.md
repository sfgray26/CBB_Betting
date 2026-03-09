# Kimi Research Mission — Fantasy Baseball 2026 Projections

**Requestor:** CBB_Betting draft assistant system
**Deadline:** Must be complete before March 23, 2026
**Output format:** CSV files dropped into `data/projections/` directory
**Priority:** CRITICAL — these replace hardcoded estimates in the live draft tool

---

## Context

We have a live fantasy baseball draft assistant for a 12-team Yahoo H2H One Win
league with 18 categories. The system currently uses hardcoded projection estimates.
Your mission is to pull REAL 2026 projection data so our recommendations are accurate
on draft day (March 23).

The system accepts CSV files in `data/projections/`. Once dropped in, they auto-load.

---

## Mission 1 — Steamer 2026 Batting Projections (HIGHEST PRIORITY)

**URL:** https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=steamer

**What to do:**
1. Navigate to that URL
2. Set "Season" = 2026, "Projection System" = Steamer
3. Download or scrape ALL batters (no PA filter — we want 300+ players)
4. Save as: `data/projections/steamer_batting_2026.csv`

**Required columns (MUST have these exact names):**
```
Name, Team, POS, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, SO, SB, CS, AVG, OBP, SLG, OPS
```

**Notes:**
- FanGraphs may call strikeouts "SO" — keep as "SO"
- We need at minimum: Name, Team, POS, PA, H, HR, R, RBI, SO, SB, CS, AVG, OPS
- Include players with as few as 50 PA — we need depth for rounds 15-23
- Player name MUST match Yahoo's display name (FanGraphs names usually do)

---

## Mission 2 — Steamer 2026 Pitching Projections (HIGHEST PRIORITY)

**URL:** https://www.fangraphs.com/projections.aspx?pos=all&stats=pit&type=steamer

**What to do:**
1. Navigate to that URL
2. Set "Season" = 2026, "Projection System" = Steamer
3. Download or scrape ALL pitchers (no IP filter)
4. Save as: `data/projections/steamer_pitching_2026.csv`

**Required columns:**
```
Name, Team, POS, W, L, ERA, G, GS, IP, H, HR, BB, SO, SV, BS, WHIP
```

**Notes:**
- "POS" should be SP, RP, or SP/RP — FanGraphs often shows this
- If FanGraphs doesn't include SV/BS in Steamer, include whatever save stats it has
- If GS is included, use it to determine SP vs RP (GS >= 10 = SP)
- We NEED at least 200 pitchers — include guys with as few as 30 IP

---

## Mission 3 — FantasyPros Yahoo ADP Consensus (HIGH PRIORITY)

**URL:** https://www.fantasypros.com/mlb/adp/overall.php

**What to do:**
1. Navigate to that URL
2. Set: Platform = Yahoo, Leagues = 12-team, Format = H2H
3. Scrape the ADP table (all 300+ players shown)
4. Save as: `data/projections/adp_yahoo_2026.csv`

**Required columns:**
```
PLAYER NAME, TEAM, POS, AVG, BEST, WORST, # TEAMS, STDEV
```

**Notes:**
- "AVG" here means average ADP (not batting average)
- Player names MUST be the full display name (e.g., "Ronald Acuña Jr." → accept "Ronald Acuna Jr.")
- We use AVG column as the primary ADP value
- STDEV column is useful for identifying consensus vs controversial picks

---

## Mission 4 — Closer Certainty Tracker (MEDIUM PRIORITY)

Research and create a file: `data/projections/closer_situations_2026.csv`

**Required columns:**
```
Team, Closer, Role (locked/likely/uncertain/committee), NSV_projection, Notes
```

**Find closer situations for ALL 30 MLB teams.**

Key questions to answer for each team:
- Who is the definitive closer? (locked = 95%+ probability)
- Are there committee situations? (e.g., Angels, Tigers)
- Any injury-related uncertainty?

**Format example:**
```csv
Team,Closer,Role,NSV_projection,Notes
CLE,Emmanuel Clase,locked,34,"Elite, no competition"
HOU,Josh Hader,locked,30,"Signed long-term"
WSH,Kyle Finnegan,likely,22,"Strong 2024 but no extension"
LAA,Carlos Estevez,uncertain,16,"Competition from Matt Moore"
```

---

## Mission 5 — 2026 Injury / Suspension / Return Timeline (MEDIUM PRIORITY)

Create: `data/projections/injury_flags_2026.csv`

**Required columns:**
```
Name, Team, Status (active/TJS_return/injury_risk/suspended),
Expected_PA_or_IP, Notes, Avoid_flag (yes/no)
```

**Key players to research:**
- Spencer Strider (ATL, SP) — TJS April 2024, return timeline for 2026?
- Kodai Senga (NYM, SP) — shoulder capsule, missed all 2024
- Shane Bieber (CLE, SP) — UCL repair, expected full return 2026?
- Eury Perez (MIA, SP) — TJS 2024, timeline?
- Carlos Rodon (NYY, SP) — flexor mass surgery
- Brandon Woodruff (MIL, SP) — shoulder surgery, return?
- Mike Trout (LAA, OF) — multiple knee surgeries, status for 2026?
- Luis Castillo (SEA, SP) — any lingering issues?
- Also: any players with PED suspensions outstanding

---

## Mission 6 — Yahoo 2026 Position Eligibility Verification (LOWER PRIORITY)

**Research task:** Verify which players have multi-position eligibility in Yahoo 2026.

Yahoo requires 20 games at a position (or 10 starts for pitchers) in the prior season
to grant eligibility. Key players to verify:

1. Shohei Ohtani — DH only, or OF eligibility?
2. Mookie Betts — SS/OF or OF only after 2025?
3. Cody Bellinger — 1B/CF eligible?
4. Jazz Chisholm — LF/2B or just 2B?
5. Bobby Witt Jr. — SS only or also 3B?
6. Elly De La Cruz — SS/3B eligible both?
7. Marcus Semien — 2B only or 2B/SS?
8. Ha-Seong Kim — SS/2B/3B all eligible?
9. Brendan Donovan — how many positions?
10. Tommy Edman — 2B/SS/CF all eligible?

Save as: `data/projections/position_eligibility_2026.csv`

**Columns:** `Name, Team, Yahoo_Positions_2026, Source_Note`

---

## Delivery Format

For each CSV file:
1. Use UTF-8 encoding
2. First row = headers (exact names as specified above)
3. One player per row
4. No blank rows between data
5. Use "FA" for free agents with no team
6. Strip accent marks from names (é → e, á → a, ó → o, ú → u, ñ → n)

**After delivering the CSVs**, run this validation:
```bash
python -m backend.fantasy_baseball.projections_loader
```

This will print how many players loaded and flag any issues.

---

## Priority Order

1. **steamer_batting_2026.csv** — CRITICAL, needed day-of-draft
2. **steamer_pitching_2026.csv** — CRITICAL
3. **adp_yahoo_2026.csv** — HIGH — enables reach/value detection
4. **closer_situations_2026.csv** — HIGH — NSV is most scarce category
5. **injury_flags_2026.csv** — MEDIUM — risk-adjusted recommendations
6. **position_eligibility_2026.csv** — LOWER — nice to have

---

## What Happens When You Deliver

Once CSVs are in `data/projections/`, the system automatically:
1. Loads real Steamer projections (replaces ~300 hardcoded estimates)
2. Merges consensus ADP for reach/value detection
3. Applies park factors from `ballpark_factors.py` (already built)
4. Applies injury risk penalties from `injury_flags_2026.csv`
5. Recalculates all z-scores with real numbers

The draft assistant on March 23 will then have real data instead of estimates.

---

*Generated by CBB_Betting draft assistant — March 9, 2026*
