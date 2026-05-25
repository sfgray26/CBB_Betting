# 2026-05-23 Codex Wave 5B Deploy Report

## Scope

Deploy current `stable/cbb-prod` workspace to Railway:

- Backend service: `Fantasy-App`
- Frontend service: `observant-benevolence`

Source commit used for this rollout:

- `17bc74b` — `chore: commit latest fantasy rollout changes`

Transient local file excluded from commit:

- `.dev_server_pid.txt`

## Deployment Evidence

Backend upload accepted:

- Service: `Fantasy-App`
- Deployment ID: `4bc73aa0-c92e-42f7-866e-093029ce5988`

Frontend upload accepted:

- Service: `observant-benevolence`
- Deployment ID: `426d4717-bd33-4440-9560-390ff6f7f4aa`

Backend build result:

- Build completed successfully
- Final image digest: `sha256:bb62a11fa17b9b039795cbdc0dbda3176c10a7d8e558ff55698cf79c5ad9a4ca`

Frontend reachability check:

- `https://observant-benevolence-production.up.railway.app/` returned HTTP `200`

## Required Post-Deploy Checks

### 1. `/health`

Endpoint:

- `https://fantasy-app-production-5079.up.railway.app/health`

Observed response:

```json
{"status":"healthy","database":"connected","scheduler":"running"}
```

Verdict:

- Backend is healthy after deploy

### 2. Budget Endpoint

Endpoint:

- `GET /api/fantasy/budget`

Observed response excerpt:

```json
{
  "budget": {
    "acquisitions_used": 0,
    "acquisitions_remaining": 8,
    "ip_accumulated": 0.0,
    "ip_minimum": 18.0,
    "ip_pace": "BEHIND",
    "week_label": "Week 10",
    "weeks_remaining": 15,
    "days_in_week_remaining": 7,
    "acquisitions_this_season": 246
  }
}
```

Verdict against requested check:

- `week_label=Week 9` -> **NO**; actual value is `Week 10`
- `ip_accumulated` present -> **YES**
- `ip_accumulated` value -> `0.0`

Important date context:

- Production response `as_of` is `2026-05-25T09:15:34-04:00`
- This is a future production-week state relative to the local session date `2026-05-23`

### 3. `lineup/current`

Endpoint:

- `GET /api/fantasy/lineup/current`

Observed result summary:

- `date`: `2026-05-25`
- `games_count`: `11`
- `no_games_today`: `false`
- `pitchers`: `[]`

Observed warning excerpt:

```json
[
  "Only 0 active pitcher slots filled -- consider streaming a SP."
]
```

Verdict:

- `pitchers` array is **empty**

### 4. Railway Backend Logs for `[flag_pitcher_starts] DIAG`

Observed log lines:

```text
2026-05-25 13:15:43,892 - backend.fantasy_baseball.daily_lineup_optimizer - INFO - [flag_pitcher_starts] DIAG: probable_pitchers_teams=26
2026-05-25 13:15:43,892 - backend.fantasy_baseball.daily_lineup_optimizer - INFO - [flag_pitcher_starts] DIAG: roster_size=24, sp_rp_p_eligible=11
2026-05-25 13:15:43,892 - backend.fantasy_baseball.daily_lineup_optimizer - INFO - [flag_pitcher_starts] DIAG: processing Eury Pérez positions=['SP', 'P'] status=None
2026-05-25 13:15:43,892 - backend.routers.fantasy - WARNING - flag_pitcher_starts failed: 'in <string>' requires string as left operand, not dict
```

Requested values:

- `roster_size = 24`
- `sp_rp_p_eligible = 11`

Interpretation:

- The optimizer sees pitcher-eligible players in the roster
- The pipeline then fails inside `flag_pitcher_starts`
- The current production symptom is not "no pitcher-eligible players"; it is a runtime type error during pitcher-start flagging

## Outcome

Successful:

- Current workspace committed
- Backend uploaded and built successfully
- Backend `/health` is healthy
- Frontend is reachable

Still broken after deploy:

- Budget check does **not** match the requested Week 9 expectation; production is returning `Week 10`
- `ip_accumulated` is `0.0`
- `lineup/current` still returns an empty `pitchers` array
- Backend logs show `flag_pitcher_starts` failing with:
  - `'in <string>' requires string as left operand, not dict`

## Recommended Next Step

Escalate back to Claude for backend remediation of the `flag_pitcher_starts` runtime type error. The deploy is healthy, but the Wave 5B pitcher issue remains unresolved in production.
