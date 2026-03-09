"""
Fantasy Baseball Dashboard — Treemendous League (Yahoo ID 72586)
12-team H2H One Win · 18 categories · Snake Draft · Keeper League

Pages:
  - Keeper Evaluator  (active: deadline Mar 20)
  - Draft Board       (active: Mar 22–23)
  - My Roster         (season view)
  - Waiver Wire       (in-season)
  - Trade Analyzer    (in-season)
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure backend imports work when running from dashboard/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.fantasy_baseball.keeper_engine import (
    KeeperEngine,
    PlayerProjection,
    CategoryValueEngine,
    soto_2026_projection,
)

st.set_page_config(
    page_title="Fantasy Baseball — Treemendous",
    page_icon="⚾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚾ Fantasy Baseball — Treemendous League")
st.caption("Yahoo ID 72586 · 12 teams · H2H One Win · 18 categories · Keeper League")

col1, col2, col3 = st.columns(3)
col1.metric("Keeper Deadline", "Mar 20 @ 3am EDT", delta=None)
col2.metric("Draft Day", "Mar 23 @ 7:30am EDT", delta=None)
col3.metric("Draft Format", "Snake · 90-sec clock", delta=None)

st.divider()

# ---------------------------------------------------------------------------
# Navigation tabs
# ---------------------------------------------------------------------------
tab_keeper, tab_draft, tab_roster, tab_waiver, tab_trade, tab_setup = st.tabs([
    "🔑 Keeper Evaluator",
    "📋 Draft Board",
    "👥 My Roster",
    "📈 Waiver Wire",
    "↔️ Trade Analyzer",
    "⚙️ Setup & Auth",
])

# ===========================================================================
# TAB: Keeper Evaluator
# ===========================================================================
with tab_keeper:
    st.header("Keeper Evaluator")
    st.info(
        "**Deadline: Fri Mar 20 @ 3:00am EDT**  \n"
        "Keeper surplus = projected z-score minus the expected value of a player "
        "drafted at the same round. Positive surplus = worth keeping."
    )

    st.subheader("League Category Reminder")
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        st.markdown("""
**Batting (9 cats)**
| Cat | Direction | Note |
|-----|-----------|------|
| R   | ✅ Higher better | |
| H   | ✅ Higher better | Standalone — rewards contact |
| HR  | ✅ Higher better | |
| RBI | ✅ Higher better | |
| K   | ❌ Lower better | Strikeouts HURT batters |
| TB  | ✅ Higher better | SLG-correlated |
| AVG | ✅ Higher better | |
| OPS | ✅ Higher better | Most valuable rate stat |
| NSB | ✅ Higher better | SB−CS · Scarcest cat |
""")
    with cat_col2:
        st.markdown("""
**Pitching (9 cats)**
| Cat  | Direction | Note |
|------|-----------|------|
| W    | ✅ Higher better | |
| L    | ❌ Lower better | Losses HURT |
| HR   | ❌ Lower better | HR allowed |
| K    | ✅ Higher better | |
| ERA  | ❌ Lower better | |
| WHIP | ❌ Lower better | |
| K/9  | ✅ Higher better | Double K value |
| QS   | ✅ Higher better | |
| NSV  | ✅ Higher better | SV−BS · Very scarce |
""")

    st.divider()

    # --- Keeper entry form ---
    st.subheader("Add Your Keeper Candidates")
    st.caption("Enter each rostered player you're considering keeping. The engine will rank them by surplus value.")

    with st.expander("Pre-loaded: Juan Soto (adjust keeper round cost)", expanded=True):
        soto = soto_2026_projection()
        soto.keeper_round_cost = st.slider(
            "What round does keeping Soto cost you?",
            min_value=1, max_value=5, value=1,
            help="In Yahoo keeper leagues, keeping a player typically costs their draft round from last year + 1"
        )

        st.markdown(f"""
**Juan Soto — 2026 Projection (Steamer/ZiPS consensus)**

| Stat | Projection | Note |
|------|-----------|------|
| PA   | {soto.pa} | Full season |
| HR   | {soto.hr} | Elite power |
| R/RBI | {soto.r} / {soto.rbi} | Top-tier run producer |
| K%   | {soto.k_bat/soto.pa:.1%} | Low K rate for a power bat |
| AVG  | {soto.avg:.3f} | |
| OPS  | {soto.ops:.3f} | Elite |
| NSB  | {soto.nsb} | Modest base stealing |
""")

    st.markdown("---")
    st.subheader("Add More Keepers Manually")
    st.info("Use the form below to add other players on your roster you're considering keeping.")

    if "extra_keepers" not in st.session_state:
        st.session_state.extra_keepers = []

    with st.form("add_keeper_form"):
        k_col1, k_col2, k_col3 = st.columns(3)
        with k_col1:
            k_name = st.text_input("Player Name", placeholder="e.g. Yordan Alvarez")
            k_team = st.text_input("MLB Team", placeholder="e.g. HOU")
            k_type = st.selectbox("Player Type", ["batter", "pitcher"])
            k_round = st.number_input("Keeper Round Cost", min_value=1, max_value=23, value=5)
        with k_col2:
            if k_type == "batter":
                k_pa = st.number_input("PA", 0, 750, 550)
                k_r = st.number_input("R", 0, 150, 70)
                k_h = st.number_input("H", 0, 250, 140)
                k_hr = st.number_input("HR", 0, 60, 20)
                k_rbi = st.number_input("RBI", 0, 150, 70)
            else:
                k_ip = st.number_input("IP", 0.0, 250.0, 170.0)
                k_w = st.number_input("W", 0, 25, 10)
                k_l = st.number_input("L", 0, 15, 7)
                k_qs = st.number_input("QS", 0, 35, 18)
                k_sv = st.number_input("SV", 0, 50, 0)
        with k_col3:
            if k_type == "batter":
                k_k = st.number_input("K (batter Ks)", 0, 250, 100)
                k_tb = st.number_input("TB", 0, 400, 220)
                k_avg = st.number_input("AVG", 0.150, 0.400, 0.265, step=0.001, format="%.3f")
                k_ops = st.number_input("OPS", 0.500, 1.200, 0.780, step=0.001, format="%.3f")
                k_nsb = st.number_input("NSB (SB-CS)", -10, 80, 5)
            else:
                k_k_p = st.number_input("K (pitcher)", 0, 350, 160)
                k_era = st.number_input("ERA", 0.0, 8.0, 4.00, step=0.01, format="%.2f")
                k_whip = st.number_input("WHIP", 0.5, 2.5, 1.25, step=0.01, format="%.2f")
                k_k9 = st.number_input("K/9", 0.0, 15.0, 8.5, step=0.1)
                k_nsv = st.number_input("NSV (SV-BS)", -10, 50, 0)
                k_hr_p = st.number_input("HR allowed", 0, 50, 18)

        submitted = st.form_submit_button("Add Keeper Candidate")
        if submitted and k_name:
            p = PlayerProjection(
                name=k_name, yahoo_player_key="manual",
                team=k_team, positions=[], player_type=k_type,
                keeper_round_cost=k_round,
            )
            if k_type == "batter":
                p.pa = k_pa; p.r = k_r; p.h = k_h; p.hr = k_hr; p.rbi = k_rbi
                p.k_bat = k_k; p.tb = k_tb; p.avg = k_avg; p.ops = k_ops; p.nsb = k_nsb
                p.slg = k_ops - 0.320  # rough approximation if OBP unknown
            else:
                p.ip = k_ip; p.w = k_w; p.l = k_l; p.qs = k_qs
                p.sv = k_sv; p.k_pit = k_k_p; p.era = k_era; p.whip = k_whip
                p.k9 = k_k9; p.nsv = k_nsv; p.hr_pit = k_hr_p
                p.bs = max(0, k_sv - k_nsv)  # infer BS from SV and NSV
            st.session_state.extra_keepers.append(p)
            st.success(f"Added {k_name}")

    # Show added keepers
    if st.session_state.extra_keepers:
        st.write(f"**{len(st.session_state.extra_keepers)} extra keepers queued**")
        for ep in st.session_state.extra_keepers:
            st.write(f"  • {ep.name} ({ep.team}) — {ep.player_type} — Round {ep.keeper_round_cost} cost")
        if st.button("Clear all extra keepers"):
            st.session_state.extra_keepers = []

    st.divider()

    # --- Run evaluation ---
    if st.button("🔍 Run Keeper Evaluation", type="primary"):
        all_keepers = [soto] + st.session_state.extra_keepers

        # Build a representative player pool for z-score context
        # These are approximate 2026 average starters for z-score baseline
        pool = _build_baseline_pool()
        pool.extend(all_keepers)

        engine = KeeperEngine(pool)
        report = engine.evaluate_roster(all_keepers)

        st.subheader("Keeper Rankings — Surplus Value Analysis")

        import pandas as pd
        rows = []
        for p in report.players:
            surplus_str = f"+{p.keeper_surplus:.2f}" if p.keeper_surplus >= 0 else f"{p.keeper_surplus:.2f}"
            verdict = "KEEP ✓" if p.keeper_surplus > 0.5 else ("BORDERLINE" if p.keeper_surplus > 0 else "RELEASE ✗")
            rows.append({
                "Player": p.name,
                "Type": p.player_type.title(),
                "Team": p.team,
                "Round Cost": p.keeper_round_cost,
                "Z-Score": f"{p.z_score:+.2f}",
                "Surplus": surplus_str,
                "Verdict": verdict,
                "Notes": " | ".join(p.notes[:2]),
            })
        df = pd.DataFrame(rows)

        def color_verdict(val):
            if "KEEP" in val:
                return "background-color: #1a472a; color: white"
            elif "BORDERLINE" in val:
                return "background-color: #7b6d00; color: white"
            else:
                return "background-color: #6b1a1a; color: white"

        st.dataframe(
            df.style.applymap(color_verdict, subset=["Verdict"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Category Contribution Breakdown")
        for p in report.players:
            if p.category_contributions:
                with st.expander(f"{p.name} — category breakdown"):
                    cat_data = [
                        {"Category": cat, "Weighted Z-Score": f"{val:+.3f}"}
                        for cat, val in sorted(
                            p.category_contributions.items(),
                            key=lambda x: abs(x[1]), reverse=True
                        )
                    ]
                    st.dataframe(pd.DataFrame(cat_data), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB: Draft Board (placeholder — Phase 2)
# ===========================================================================
with tab_draft:
    st.header("Draft Board")
    st.warning(
        "**Draft: Mon Mar 23 @ 7:30am EDT · 90-second pick clock**  \n"
        "Draft board will be populated after keeper decisions are finalized. "
        "Live draft assistant launches Mar 22."
    )
    st.markdown("""
### What the Draft Assistant Will Do
- Real-time pick recommendations (< 10 sec, powered by local Qwen model)
- Tracks all picks as they happen — updates available pool instantly
- Shows your current roster balance across all 18 categories
- Flags reach alerts when others overdraft a player
- Snake draft logic: knows your next pick position

### Draft Strategy Framework (Treemendous)
| Round | Target | Rationale |
|-------|--------|-----------|
| 1–3   | Elite multi-cat hitters | HR + R + RBI + TB + AVG/OPS — hits 5+ cats at once |
| 4–6   | Ace SPs | W + K + K/9 + QS + ERA + WHIP — 6 pitching cats |
| 7–9   | **SB assets** | NSB is scarce — stock up mid-rounds |
| 10–13 | Solid SP depth | Minimize L category exposure |
| 14–17 | Closer pipeline | NSV is very scarce; take 2–3 closers |
| 18–23 | Streaming SPs, bench | High K/9, low L risk |

### Contact + Power > Pure Power (K is negative)
High-K sluggers (25+ HR, 30%+ K) are double-penalized: K count hurts + drags AVG/OPS.
**Target**: Yordan Alvarez, Freddie Freeman, Paul Goldschmidt, Luis Arraez types —
power WITH contact. Avoid Brandon Drury-types who only contribute HR/RBI.
""")

# ===========================================================================
# TAB: My Roster
# ===========================================================================
with tab_roster:
    st.header("My Roster")
    yahoo_ready = bool(os.getenv("YAHOO_REFRESH_TOKEN"))
    if not yahoo_ready:
        st.warning("Yahoo API not authenticated yet. Go to **Setup & Auth** tab to connect.")
    else:
        st.info("Click below to sync your current roster from Yahoo.")
        if st.button("Sync Roster from Yahoo"):
            try:
                from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
                client = YahooFantasyClient()
                with st.spinner("Fetching roster..."):
                    roster = client.get_roster()
                st.success(f"Fetched {len(roster)} players")
                import pandas as pd
                df = pd.DataFrame(roster)
                st.dataframe(df[["name", "team", "positions", "status"]], use_container_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")

# ===========================================================================
# TAB: Waiver Wire
# ===========================================================================
with tab_waiver:
    st.header("Waiver Wire Intelligence")
    st.info("In-season feature — available after March 24 draft.")
    st.markdown("""
**Waiver Rules — Treemendous**
- Type: Continual rolling list
- Wait time: 1 day
- Max acquisitions/week: **8** (hard cap — prioritize wisely)
- Injured players cannot go directly to IL slot from waivers

**What the waiver engine will do:**
- Rank free agents by: (a) category need vs. opponent, (b) rest-of-season value
- Highlight streaming SPs with good matchups this week
- Track your waiver priority position
- Alert when a valuable player is dropped by another team
""")

# ===========================================================================
# TAB: Trade Analyzer
# ===========================================================================
with tab_trade:
    st.header("Trade Analyzer")
    st.info("In-season feature — trade deadline August 6, 2026.")
    st.markdown("""
**How it works:**
1. Enter the proposed trade (players you give, players you receive)
2. Engine computes net category delta across all 18 categories
3. Shows which categories improve, which weaken
4. Toggle: Win-now mode vs. rebuild mode
5. ROS (rest-of-season) projections used for all valuations

**Trade Deadline: August 6, 2026**
""")

# ===========================================================================
# TAB: Setup & Auth
# ===========================================================================
with tab_setup:
    st.header("Yahoo API Setup")

    yahoo_client_id = os.getenv("YAHOO_CLIENT_ID", "")
    yahoo_refresh = os.getenv("YAHOO_REFRESH_TOKEN", "")

    if yahoo_client_id:
        st.success(f"Client ID loaded: {yahoo_client_id[:20]}...")
    else:
        st.error("YAHOO_CLIENT_ID not found in .env")

    if yahoo_refresh:
        st.success("Refresh token found — API is ready")
    else:
        st.warning("No refresh token yet — complete OAuth flow below")

    st.markdown("---")
    st.subheader("One-Time Authorization")
    st.markdown("""
Run this command in your terminal to complete OAuth setup:

```bash
cd /home/user/CBB_Betting
python -m backend.fantasy_baseball.yahoo_client --auth
```

This will:
1. Open a browser to Yahoo's authorization page
2. You authorize the app
3. Yahoo gives you a 6-digit code
4. Paste it back in the terminal
5. Refresh token is saved to `.env` automatically

After auth, click **Sync Roster** in the My Roster tab to verify.
""")

    st.markdown("---")
    st.subheader("Security Reminder")
    st.error(
        "⚠️ Your Yahoo API credentials were shared in a chat session. "
        "Please rotate them at https://developer.yahoo.com/apps/ "
        "and update your .env file with the new credentials."
    )


# ---------------------------------------------------------------------------
# Helper: baseline player pool for z-score context
# ---------------------------------------------------------------------------

def _build_baseline_pool() -> list[PlayerProjection]:
    """
    Approximate 2026 baseline player pool for z-score calibration.
    These represent typical drafted starters in a 12-team league.
    Weights are tuned for Treemendous category set.
    """
    batters = [
        # name, team, pa, r, h, hr, rbi, k_bat, tb, avg, ops, nsb
        ("Ronald Acuna Jr.", "ATL", 680, 115, 185, 35, 95, 155, 340, 0.302, 0.975, 55),
        ("Freddie Freeman", "LAD", 660, 100, 190, 28, 100, 120, 310, 0.300, 0.910, 10),
        ("Yordan Alvarez", "HOU", 620, 95, 165, 37, 108, 140, 340, 0.295, 0.970, 3),
        ("Mookie Betts", "LAD", 650, 110, 180, 30, 90, 135, 315, 0.295, 0.930, 15),
        ("Julio Rodriguez", "SEA", 650, 95, 175, 30, 90, 160, 300, 0.285, 0.870, 28),
        ("Corey Seager", "TEX", 580, 90, 165, 30, 95, 110, 295, 0.295, 0.920, 5),
        ("José Ramirez", "CLE", 650, 100, 175, 28, 100, 85, 300, 0.285, 0.880, 25),
        ("Elly De La Cruz", "CIN", 620, 95, 160, 25, 85, 200, 275, 0.265, 0.830, 50),
        ("Gunnar Henderson", "BAL", 640, 100, 170, 32, 95, 165, 310, 0.278, 0.890, 18),
        ("Juan Soto", "NYY", 680, 105, 172, 38, 100, 128, 330, 0.288, 0.977, 7),
        ("Bobby Witt Jr.", "KC", 660, 105, 185, 28, 95, 130, 300, 0.295, 0.890, 40),
        ("Adley Rutschman", "BAL", 580, 80, 155, 18, 80, 95, 255, 0.275, 0.840, 5),
        ("Trea Turner", "PHI", 620, 95, 175, 22, 80, 130, 285, 0.295, 0.860, 28),
        ("Francisco Lindor", "NYM", 640, 95, 165, 28, 95, 145, 295, 0.275, 0.860, 22),
        ("Marcus Semien", "TEX", 660, 100, 170, 25, 88, 135, 285, 0.270, 0.840, 12),
        ("William Contreras", "MIL", 560, 75, 150, 22, 80, 105, 250, 0.275, 0.840, 5),
        ("Bryce Harper", "PHI", 580, 90, 160, 30, 95, 120, 295, 0.285, 0.930, 8),
        ("Pete Alonso", "NYM", 620, 80, 150, 38, 105, 155, 305, 0.255, 0.870, 2),
        ("Spencer Steer", "CIN", 580, 78, 148, 22, 82, 130, 252, 0.265, 0.800, 12),
        ("Jazz Chisholm", "NYY", 560, 82, 145, 22, 78, 155, 250, 0.262, 0.810, 22),
    ]
    pitchers = [
        # name, team, ip, w, l, sv, bs, qs, k, era, whip, k9, hr, nsv
        ("Gerrit Cole", "NYY", 185, 14, 6, 0, 0, 22, 225, 2.80, 1.00, 10.9, 18, 0),
        ("Spencer Strider", "ATL", 175, 14, 5, 0, 0, 21, 240, 2.70, 0.95, 12.3, 16, 0),
        ("Logan Webb", "SF", 200, 14, 7, 0, 0, 24, 195, 3.10, 1.08, 8.8, 17, 0),
        ("Zack Wheeler", "PHI", 195, 14, 6, 0, 0, 24, 215, 3.00, 1.05, 9.9, 19, 0),
        ("Paul Skenes", "PIT", 170, 12, 7, 0, 0, 20, 220, 2.90, 1.02, 11.6, 16, 0),
        ("Corbin Burnes", "BAL", 190, 13, 7, 0, 0, 22, 205, 3.10, 1.06, 9.7, 18, 0),
        ("Emmanuel Clase", "CLE", 70, 4, 2, 38, 5, 0, 82, 2.20, 0.90, 10.5, 5, 33),
        ("Josh Hader", "HOU", 65, 3, 3, 35, 6, 0, 88, 2.40, 0.95, 12.2, 4, 29),
        ("Edwin Diaz", "NYM", 65, 3, 3, 32, 5, 0, 92, 2.50, 0.95, 12.7, 4, 27),
        ("Taijuan Walker", "PHI", 160, 10, 8, 0, 0, 18, 150, 3.80, 1.22, 8.4, 20, 0),
        ("Dylan Cease", "SD", 180, 12, 8, 0, 0, 20, 200, 3.40, 1.15, 10.0, 19, 0),
        ("Framber Valdez", "HOU", 195, 13, 8, 0, 0, 23, 175, 3.25, 1.18, 8.1, 16, 0),
        ("Kevin Gausman", "TOR", 185, 12, 8, 0, 0, 22, 195, 3.20, 1.05, 9.5, 20, 0),
        ("Shane Bieber", "CLE", 175, 12, 7, 0, 0, 21, 190, 3.15, 1.05, 9.8, 18, 0),
        ("Camilo Doval", "SF", 65, 3, 4, 30, 7, 0, 72, 2.80, 1.05, 9.9, 5, 23),
    ]

    pool = []
    for b in batters:
        (name, team, pa, r, h, hr, rbi, k_bat, tb, avg, ops, nsb) = b
        p = PlayerProjection(
            name=name, yahoo_player_key="", team=team,
            positions=["OF"], player_type="batter",
            pa=pa, r=r, h=h, hr=hr, rbi=rbi, k_bat=k_bat,
            tb=tb, avg=avg, ops=ops, nsb=nsb,
        )
        p.slg = ops - 0.330
        pool.append(p)
    for pi in pitchers:
        (name, team, ip, w, l, sv, bs, qs, k, era, whip, k9, hr_p, nsv) = pi
        p = PlayerProjection(
            name=name, yahoo_player_key="", team=team,
            positions=["SP"] if sv == 0 else ["RP"], player_type="pitcher",
            ip=ip, w=w, l=l, sv=sv, bs=bs, qs=qs, k_pit=k,
            era=era, whip=whip, k9=k9, hr_pit=hr_p, nsv=nsv,
        )
        pool.append(p)
    return pool
