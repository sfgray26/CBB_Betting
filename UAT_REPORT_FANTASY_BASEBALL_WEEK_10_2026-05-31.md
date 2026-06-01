# CBB Edge — Fantasy Baseball Modules UAT Report

**Reviewer perspective:** Fantasy baseball expert / multi-time league winner
**Build reviewed:** observant-benevolence-production.up.railway.app — War Room suite (Week 10, data as of 12:09–12:11 PM EDT, 2026‑05‑31)
**Scope:** `/war-room` (matchup), `/war-room/roster`, `/war-room/waiver`, `/war-room/streaming`, `/war-room/budget`, `/war-room/preview`
**Verdict:** The tools have a strong skeleton (z-score valuation, category win-probability framing, constraint tracking) but are undermined by data-integrity bugs, contradictory numbers, and missing decision logic. In their current state an elite manager could not trust them to set a lineup or make adds. Several modules are effectively non-functional.

Severity legend: **P0** = blocks core use / shows wrong info a manager would act on · **P1** = materially misleading or major gap · **P2** = polish/UX.

---

## 1. Lineup Optimizer (`/war-room/roster` → "Optimize Lineup")

This is the most broken high-value feature and the one the user flagged. Multiple defects:

**P0 — `proxy_projection` fallback assigns a flat 58.0 score to half the lineup.** Dillon Dingler, Luke Keaschall, Vinnie Pasquantino, Jordan Walker, Sam Antonacci, Pete Alonso, and Cristopher Sánchez all score exactly **58.0 (proxy_projection)**. The optimizer is silently substituting a constant placeholder whenever it lacks a real projection. This means the "optimized" lineup is not optimized — it's sorting noise. *Fix: never emit a flat fallback into a ranking. If a real projection is missing, either compute from the season/ROS stats already shown on the roster card or flag the player as "no projection — excluded from optimization" rather than injecting a tie value.*

**P0 — The optimizer benches better players than it starts.** It starts **Jordan Walker at OF (58.0)** while benching **Gavin Williams (84.6)** and recommends benching **Juan Soto with a score of 20.0** — Soto is the best hitter on the roster (.300/.974 OPS, 12 HR). Any optimizer that benches a .974‑OPS hitter is broken in the eyes of a manager and will destroy trust instantly. *Fix: root cause is almost certainly the 58.0 fallback colliding with real scores; once scoring is fixed, add a regression test asserting that no player with a top-3 roster OPS/wOBA is benched without an injury/no-game reason.*

**P0 — Pitchers are being ranked head-to-head with hitters on one scale.** SP/RP show scores of 96–99 while hitters cap near 87, purely because the scoring scales differ, not because pitchers are "better." This biases roster construction. *Fix: normalize hitter and pitcher scores to a common unit (z-score vs. position, or projected category points) before ranking.*

**P0 — "No Game" everywhere but the optimizer still produces a daily lineup.** Every player shows "No Game," yet the tool outputs a lineup "for 2026‑05‑31." If the schedule feed is empty, the optimizer should not be silently optimizing against zero games. *Fix: gate optimization on a valid schedule; if no games are loaded, show "No games detected for this date" instead of a confident lineup.*

**P1 — Self-contradicting date/data banner.** Header literally reads: *"Optimized lineup for 2026‑05‑31 (Note: Data from 2026‑05‑31, not requested 2026‑05‑31)."* This is a debug string leaking to users and indicates a date-mismatch code path that is firing even when the dates are identical. *Fix: remove the debug note; correct the comparison logic that thinks the requested date ≠ data date.*

**P1 — "APPLY ALL 19 MOVES" with no diff/confirmation.** A one-click 19-move apply with no preview of what changes, no warning that it benches Soto, and no undo is dangerous given the league's 8-move weekly cap. *Fix: show a confirmation diff (player → new slot) and a count of acquisitions/moves consumed before applying.*

**P2 — Score provenance tags (`player_scores`, `proxy_projection`) are exposed raw.** Useful internally, confusing to managers. Replace with a confidence indicator or hide.

---

## 2. My Roster — matchup & season aggregates (`/war-room/roster`)

**P0 — "THIS WEEK · VS OPPONENT" shows all zeros, 0W‑0L‑18T, 0% win prob.** The weekly matchup panel on the roster page is completely empty/tied across all 18 categories while the War Room page (same week, same opponent) shows live 9‑7 scoring. The two pages are reading different (or stale) data sources. *Fix: unify the matchup data source between `/war-room` and `/war-room/roster`; the roster page is clearly not receiving the live scoreboard feed.*

**P1 — Team season AVG of .237 is not credible** given the roster contains Soto (.300), Walker (.290), Antonacci (.284), Waldschmidt (.296). A real team built from these hitters would post a markedly higher aggregate AVG. This points to a rate-stat aggregation bug (likely averaging-of-averages or including 0.000 placeholder rows from non-hitters/"No Game" players). *Fix: compute AVG as ΣH/ΣAB (and OPS via component aggregation), excluding players with no AB; add a unit test against a known roster.*

**P1 — Stat-mapping / identity integrity risk (see §6 data accuracy).** Cross-check flagged that ownership and per-player lines may be misattributed or stale.

**P2 — "Move to…" menus and slot eligibility look correct** (e.g., Crochet/Díaz/Murakami on IL, eligibility lists match real positions) — this part is good and should be preserved.

---

## 3. Waiver Wire (`/war-room/waiver`)

The user's core complaint is correct and is the #1 functional gap here.

**P0 — No drop recommendation / add-drop pairing.** Every suggested add (Soderstrom, Caballero, Nimmo, etc.) shows a Match Score and "fit for your gaps" but never says **who to drop**, nor whether the add is even an upgrade over the worst rosterable player at that position. In a league with a hard 8-move weekly cap and a full 22-man roster, an add is meaningless without a corresponding drop. *Fix: for each suggested add, compute and display the recommended drop (lowest ROS-value rosterable player at a compatible slot), the net category delta, and a "blocked — no worthwhile drop" state when applicable.*

**P1 — "Loading recommendations…" persists at top of page** even after the list renders, suggesting an unresolved async state. *Fix: clear the loading flag on data arrival.*

**P1 — Tags are unexplained and sometimes contradictory.** Players are tagged `BUY_LOW`, `BREAKOUT`, `HOT`/`COLD`, `PARK+/-`, yet Daulton Varsho is simultaneously `COLD ▼` and `BUY_LOW`, and Jake McCarthy is `COLD ▼ BUY_LOW PARK+`. There's no legend and no logic shown for why a cold player is a buy-low. *Fix: add a tooltip/legend defining each tag and the threshold that triggered it.*

**P1 — Ownership context missing for ranking.** José Caballero (0% owned) outranks Brandon Nimmo (26% owned) purely on z-score. A real waiver tool should weight availability and likelihood of being claimed. *Fix: surface "realistically available" filtering and show FAAB/claim-priority guidance.*

**P2 — "⚠️ Small Sample" flag is good** — keep it; consider extending it to hitters with low AB, not just relievers.

---

## 4. Streaming Station (`/war-room/streaming`)

**P0 — It's a duplicate of the Waiver Wire, not a streaming tool.** The "NEED" scores are identical to the Waiver Wire season z-scores (Caballero 29.2, Soderstrom 26.1, Nimmo 25.7…). Streaming is fundamentally about **upcoming schedule** — two-start pitcher weeks, soft matchups, park factors over the next 3–7 days, off-day coverage. This page ranks by season-long value, which is the wrong axis entirely. *Fix: rebuild ranking on next-N-days projected category contribution (opponent quality, park, # of games/starts), not season z-score.*

**P1 — "2‑Start SPs Only" filter is incoherent with the data.** The unfiltered list is dominated by hitters and relievers; a 2-start filter only makes sense if start schedules are loaded — which, given §2's "No Game" everywhere, they are not. *Fix: wire the filter to real probable-pitcher/schedule data; hide it if schedule data is unavailable.*

**P2 — "3 hidden" with "Hide Owned Players" on** is fine, but show what's hidden on hover.

---

## 5. Budget (`/war-room/budget`)

**P1 — Internal inconsistency on Innings Pitched.** This page shows **"Innings Pitched — PENDING / Yahoo stats syncing… / min 18 IP"**, while the Roster page confidently shows **"IP PACE — AHEAD — 30.1 / 18 IP."** One says data isn't synced; the other reports a precise pace. *Fix: single source of truth for IP; if syncing, both pages should say pending.*

**P1 — "SEASON ADDS 272" is implausible / unlabeled.** With 8/8 weekly acquisitions remaining and a 15-week season left, 272 season adds needs definition (is it a cap? a count used? league-wide?). As shown it reads like a bug or an unlabeled field. *Fix: label the metric and verify the calculation.*

**P2 — Otherwise this page is clean and useful** (acquisitions 0/8, IL 3/3 Full, Week 10, 15 weeks left). The IL "Full" warning is genuinely helpful.

---

## 6. Weekly Preview (`/war-room/preview`) — **most broken module**

**P0 — Opponent is "Unknown" and the category table is empty.** "NEXT OPPONENT: Unknown," "SCHEDULE ADVANTAGE: My Team 0 games / Unknown 0 games," and the CATEGORY PROJECTIONS table has headers but zero rows. The page has no usable content. *Fix: the next-opponent and schedule lookups for Week 11 are failing — wire them to the same scheduling/standings source the War Room uses.*

**P0 — Logically contradictory headline numbers.** It shows **"PROJECTED WIN% 100%"** while simultaneously stating **"Projected to lose K (0% win rate)."** You cannot be 100% to win the matchup while projected to lose a category. The 100% is a fallback-against-an-unknown-opponent artifact. *Fix: when opponent is unknown, suppress the win% entirely rather than defaulting to 100%.*

---

## 7. Data Accuracy — cross-check vs. ESPN (Josh Jung, TEX 3B)

**P0/P1 — Player stats and ownership appear stale or misattributed.** Per ESPN's 2026 game log, Josh Jung's actual season line is **.307 AVG, 6 HR, 22 RBI, 24 R, .839 OPS, 59 H** (192 AB). CBB Edge lists Jung on the Waiver Wire as a **`BUY_LOW` at 4% owned** with a season value of +21.8z. A third baseman hitting .307 with a .839 OPS would not be 4% owned in any real league — this strongly suggests the ownership feed and/or the per-player stat snapshot is stale or mis-keyed to the wrong player ID. *Fix: (a) verify the player-ID join between the stats provider and the ownership provider; (b) add a sanity monitor that flags "high performer at very low ownership" as a likely data-integrity error; (c) confirm the stat snapshot date matches the displayed "data as of" timestamp.* Recommend extending this cross-check to a sample of 10–15 rostered/available players before release.

---

## Cross-cutting issues (apply to all modules)

The single biggest root cause is a **broken or empty schedule/games feed** ("No Game" on every roster player) cascading into the Optimizer, Streaming, and Preview. Fixing the schedule ingestion will likely resolve several P0s at once. The second systemic issue is **silent fallbacks** — flat 58.0 scores, default 100% win%, "Unknown" opponent — where the app fabricates confident-looking output instead of surfacing missing data; every fallback should be replaced with an explicit "data unavailable" state. Third, there is **no single source of truth**: the same week's matchup, IP pace, and player stats differ between pages, so these reads should be consolidated. Finally, **debug strings are leaking to users** (the date "Note" in the optimizer) and should be stripped from the production build.

## Suggested fix priority for Claude Code

Start with the schedule/games feed (unblocks Optimizer, Streaming, Preview), then the Optimizer scoring fallback and hitter/pitcher normalization, then the Waiver add-drop pairing, then the roster rate-stat aggregation, then the Preview unknown-opponent handling, and finally the Streaming rebuild onto a schedule-based ranking. The data-accuracy/ID-join investigation in §7 should run in parallel since it affects every module.

---

One note before any further action: I reviewed and reported only — I did **not** click "Apply," "Apply All 19 Moves," or "Re-run," since those would mutate your actual lineup/roster. If you'd like, I can re-test the Optimizer's "Apply" behavior or capture screenshots of any specific module to attach to this report, but I'll hold off on anything that changes your team until you confirm.
