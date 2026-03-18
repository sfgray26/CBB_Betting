"""
Tournament Live — Real-time NCAA Tournament scores and bracket results.

Data source: BallDontLie GOAT API
Requires: BALLDONTLIE_API_KEY in .env

Features:
- Live/completed tournament game scores by date
- Official bracket results (all rounds, from /ncaab/v1/bracket)
- Auto-advance winners: updates bracket_2026.json as games complete
- Player performance during the tournament
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import streamlit as st
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(
        Path(__file__).resolve().parent.parent.parent / ".env",
        encoding="utf-8",
        override=False,
    )
except ImportError:
    pass

st.set_page_config(
    page_title="Tournament Live | CBB Edge",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bracket_2026.json"
REGIONS = ["east", "west", "south", "midwest"]
ROUND_NAMES = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
               4: "Elite 8", 5: "Final Four", 6: "Championship", 7: "First Four"}

# ---------------------------------------------------------------------------
# BDL client init — no cache_resource; Railway injects env vars directly
# ---------------------------------------------------------------------------
def get_client():
    from backend.services.balldontlie import BallDontLieClient
    api_key = os.environ.get("BALLDONTLIE_API_KEY", "").strip()
    if not api_key:
        st.error("BALLDONTLIE_API_KEY not found in environment.")
        st.code(f"Env vars with 'BALL': {[k for k in os.environ if 'BALL' in k.upper()]}")
        st.info("Set BALLDONTLIE_API_KEY in Railway dashboard (or .env locally) and redeploy.")
        st.stop()
    return BallDontLieClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def status_badge(status: str) -> str:
    s = (status or "").lower()
    if s in ("final", "completed", "complete"):
        return "✅"
    if s in ("in progress", "live", "halftime"):
        return "🔴 LIVE"
    return "🕐"


def score_line(game: dict) -> str:
    home = (game.get("home_team") or {}).get("name", "?")
    away = (game.get("visitor_team") or game.get("away_team") or {}).get("name", "?")
    hs = game.get("home_team_score")
    as_ = game.get("visitor_team_score") or game.get("away_team_score")
    status = game.get("status", "")
    badge = status_badge(status)
    if hs is not None and as_ is not None:
        return f"{badge} **{away} {as_}** @ **{home} {hs}**"
    return f"{badge} {away} @ {home}"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

st.title("🏀 2026 NCAA Tournament — Live Tracker")
st.caption("Powered by BallDontLie GOAT API · Subscription valid through April 7, 2026")

tab_today, tab_bracket, tab_stats, tab_debug = st.tabs(
    ["Today's Games", "Full Bracket Results", "Team Stats", "API Debug"]
)

# ---- Tab 1: Today's Games ----
with tab_today:
    col_date, col_refresh = st.columns([3, 1])
    with col_date:
        target_date = st.date_input("Date", value=date.today(), key="live_date")
    with col_refresh:
        st.write("")
        refresh_clicked = st.button("Refresh", use_container_width=True)

    client = get_client()

    @st.cache_data(ttl=60, show_spinner="Fetching games...")
    def fetch_games(d: str) -> List[dict]:
        return client.get_live_tournament_games(d)

    if refresh_clicked:
        st.cache_data.clear()

    games = fetch_games(target_date.isoformat())

    if not games:
        st.info(f"No tournament games found for {target_date}. "
                f"The tournament runs March 18 – April 7, 2026.")
    else:
        # Group by status
        live = [g for g in games if "live" in (g.get("status") or "").lower()
                or "progress" in (g.get("status") or "").lower()]
        completed = [g for g in games if (g.get("status") or "").lower()
                     in ("final", "completed", "complete")]
        upcoming = [g for g in games if g not in live and g not in completed]

        if live:
            st.subheader(f"🔴 Live ({len(live)})")
            for g in live:
                st.markdown(score_line(g))
                period = g.get("period") or g.get("time") or ""
                if period:
                    st.caption(f"  Period/Time: {period}")

        if completed:
            st.subheader(f"✅ Final ({len(completed)})")
            for g in completed:
                st.markdown(score_line(g))

        if upcoming:
            st.subheader(f"🕐 Upcoming ({len(upcoming)})")
            for g in upcoming:
                home = (g.get("home_team") or {}).get("name", "?")
                away = (g.get("visitor_team") or g.get("away_team") or {}).get("name", "?")
                tip = g.get("datetime") or g.get("date") or ""
                st.write(f"  {away} @ {home}  {tip}")

    # Auto-advance button
    st.divider()
    if st.button("Auto-Advance Bracket from Completed Games",
                 help="Updates bracket_2026.json with actual winners"):
        completed_games = [g for g in games
                           if (g.get("status") or "").lower()
                           in ("final", "completed", "complete")]
        if not completed_games:
            st.info("No completed games today to advance.")
        else:
            with open(BRACKET_PATH, encoding="utf-8") as f:
                raw = json.load(f)

            advanced = 0
            for g in completed_games:
                hs = g.get("home_team_score")
                as_ = g.get("visitor_team_score") or g.get("away_team_score")
                home_team = (g.get("home_team") or {}).get("name", "")
                away_team = (g.get("visitor_team") or g.get("away_team") or {}).get("name", "")
                if hs is None or as_ is None:
                    continue
                winner_name = home_team if int(hs) > int(as_) else away_team
                loser_name = away_team if int(hs) > int(as_) else home_team
                # Mark the loser as eliminated in the JSON
                for region in REGIONS:
                    for team in raw.get(region, []):
                        if team.get("name", "").lower() == loser_name.lower():
                            team["eliminated"] = True
                            advanced += 1

            if advanced:
                with open(BRACKET_PATH, "w", encoding="utf-8") as f:
                    json.dump(raw, f, indent=2, ensure_ascii=False)
                st.success(f"Marked {advanced} eliminations in bracket_2026.json")
            else:
                st.info("No matching teams found in bracket JSON to eliminate.")


# ---- Tab 2: Full Bracket Results ----
with tab_bracket:
    st.subheader("Official NCAA Tournament Bracket")

    season_year = st.number_input("Season", value=2025, min_value=2020, max_value=2030,
                                  help="2025 = 2025-26 season")
    round_filter = st.selectbox(
        "Round",
        options=[0] + list(ROUND_NAMES.keys()),
        format_func=lambda x: "All Rounds" if x == 0 else ROUND_NAMES[x],
    )

    if st.button("Load Bracket Results", type="primary"):
        client = get_client()
        with st.spinner("Fetching bracket from BallDontLie..."):
            try:
                if round_filter == 0:
                    bracket_data = client.get_full_bracket(season=int(season_year))
                    total_games = sum(len(v) for v in bracket_data.values())
                    st.success(f"Loaded {total_games} bracket entries across all rounds")

                    for round_key, games_list in bracket_data.items():
                        if not games_list:
                            continue
                        with st.expander(f"{round_key.upper()} ({len(games_list)} games)",
                                         expanded=(round_key == "r64")):
                            for g in games_list:
                                st.markdown(score_line(g))
                else:
                    games_list = client.get_bracket(
                        season=int(season_year), round_id=round_filter
                    )
                    st.success(f"{ROUND_NAMES[round_filter]}: {len(games_list)} games")
                    for g in games_list:
                        st.markdown(score_line(g))

            except Exception as exc:
                st.error(f"Error fetching bracket: {exc}")
                if "401" in str(exc) or "403" in str(exc):
                    st.info("This endpoint requires GOAT tier. Verify your subscription.")


# ---- Tab 3: Team Season Stats ----
with tab_stats:
    st.subheader("Team Season Stats")
    st.caption("Source: BallDontLie /ncaab/v1/team_season_stats (2025-26 season)")

    if st.button("Load Team Stats", type="primary"):
        client = get_client()
        with st.spinner("Fetching team season stats..."):
            try:
                import pandas as pd
                stats = client.get_team_season_stats()
                if not stats:
                    st.info("No stats returned.")
                else:
                    rows = []
                    for row in stats:
                        team = row.get("team", {})
                        rows.append({
                            "Team": team.get("name", ""),
                            "Conference": team.get("conference", ""),
                            "GP": row.get("games_played"),
                            "PPG": row.get("pts"),
                            "FG%": row.get("fg_pct"),
                            "3P%": row.get("fg3_pct"),
                            "FT%": row.get("ft_pct"),
                            "REB": row.get("reb"),
                            "AST": row.get("ast"),
                            "STL": row.get("stl"),
                            "BLK": row.get("blk"),
                            "TOV": row.get("turnover"),
                        })
                    df = pd.DataFrame(rows)
                    df = df.sort_values("PPG", ascending=False, na_position="last")
                    st.dataframe(df, use_container_width=True, height=500)
                    st.download_button(
                        "Download CSV", df.to_csv(index=False),
                        file_name="ncaab_team_stats_2026.csv", mime="text/csv"
                    )
            except Exception as exc:
                st.error(f"Error: {exc}")


# ---- Tab 4: API Debug ----
with tab_debug:
    st.subheader("API Debug & Connectivity Test")
    st.caption("Test BallDontLie API connectivity and inspect raw responses")

    endpoint = st.selectbox("Endpoint", [
        "/ncaab/v1/games",
        "/ncaab/v1/bracket",
        "/ncaab/v1/odds",
        "/ncaab/v1/team_season_stats",
        "/ncaab/v1/teams",
        "/ncaab/v1/standings",
    ])

    extra_params = st.text_input("Extra query params (JSON)",
                                  value='{"season": 2025, "per_page": 5}',
                                  help='e.g. {"dates": "2026-03-20"} for odds')

    if st.button("Test Endpoint"):
        import requests as _requests
        api_key = os.getenv("BALLDONTLIE_API_KEY", "")
        if not api_key:
            st.error("BALLDONTLIE_API_KEY not set")
        else:
            try:
                params = json.loads(extra_params) if extra_params.strip() else {}
                url = f"https://api.balldontlie.io{endpoint}"
                resp = _requests.get(
                    url,
                    headers={"Authorization": api_key},
                    params=params,
                    timeout=15,
                )
                st.write(f"Status: **{resp.status_code}**")
                data = resp.json()
                meta = data.get("meta", {})
                records = data.get("data", [])
                if meta:
                    st.write(f"Meta: {meta}")
                st.write(f"Records returned: {len(records)}")
                if records:
                    st.json(records[:3])
            except json.JSONDecodeError:
                st.error("Invalid JSON in extra params")
            except Exception as exc:
                st.error(f"Request failed: {exc}")
