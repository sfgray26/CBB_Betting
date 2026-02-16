"""
Streamlit Dashboard for CBB Edge Analyzer
Simple, functional UI for monitoring predictions and performance
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY_USER1", "")

st.set_page_config(
    page_title="CBB Edge Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .big-metric { font-size: 24px; font-weight: bold; }
    .positive { color: green; }
    .negative { color: red; }
    .warning { color: orange; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# API HELPERS
# ==============================================================================

def _headers():
    return {"X-API-Key": st.session_state.get("api_key", API_KEY)}


def make_request(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", headers=_headers(), params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def make_post_request(endpoint: str, payload: dict):
    try:
        r = requests.post(
            f"{API_URL}{endpoint}",
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"API Error {e.response.status_code}: {detail}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def make_put_request(endpoint: str, payload: dict):
    try:
        r = requests.put(
            f"{API_URL}{endpoint}",
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"API Error {e.response.status_code}: {detail}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("🏀 CBB Edge")
    st.caption("Version 7 Betting Framework")

    if not API_KEY:
        key_input = st.text_input("API Key", type="password", key="api_key_input")
        if key_input:
            st.session_state["api_key"] = key_input
            st.success("Key set for this session")
    else:
        st.session_state.setdefault("api_key", API_KEY)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🎯 Today's Bets", "📋 Bet Log"],
    )
    st.caption("See sidebar pages for Performance, CLV, History, Calibration & Alerts.")

    st.markdown("---")

    st.subheader("Quick Stats")
    perf = make_request("/api/performance/summary")
    if perf and perf.get("total_bets", 0) > 0:
        st.metric("Total Bets", perf["total_bets"])
        st.metric("Win Rate", f"{perf.get('win_rate', 0):.1%}")
        roi = perf.get("roi", 0)
        st.metric("ROI", f"{roi:.1%}", delta=f"{'↑' if roi > 0 else '↓'}")
        clv = perf.get("mean_clv", 0)
        st.metric("Mean CLV", f"{clv:.2%}", delta=perf.get("status", ""))
    else:
        st.info("No settled bets yet")


# ==============================================================================
# DASHBOARD PAGE
# ==============================================================================

if page == "📊 Dashboard":
    st.title("CBB Edge Analyzer")

    col1, col2, col3, col4 = st.columns(4)
    if perf and perf.get("total_bets", 0) > 0:
        with col1:
            st.metric("Total Bets", perf["total_bets"])
        with col2:
            st.metric("Win Rate", f"{perf['win_rate']:.1%}")
        with col3:
            roi = perf["roi"]
            st.metric("ROI", f"{roi:.1%}", delta="positive" if roi > 0 else "negative")
        with col4:
            st.metric("Mean CLV", f"{perf['mean_clv']:.2%}")

        st.markdown("---")
        status = perf.get("status", "UNKNOWN")
        if status == "HEALTHY":
            st.success("System Status: HEALTHY (CLV > 0.5%)")
        elif status == "WARNING":
            st.warning("System Status: WARNING (CLV near zero)")
        else:
            st.error("System Status: STOP BETTING (CLV negative)")
    else:
        st.info("No settled bets yet — predictions and paper trades will appear here once games complete.")

    st.markdown("---")
    st.subheader("Recent Bet Recommendations (last 7 days)")
    bets_data = make_request("/api/predictions/bets", {"days": 7})
    if bets_data and bets_data.get("bets"):
        st.dataframe(pd.DataFrame(bets_data["bets"]), use_container_width=True)
    else:
        st.info("No bets recommended in the last 7 days.")


# ==============================================================================
# TODAY'S BETS PAGE
# ==============================================================================

elif page == "🎯 Today's Bets":
    st.title("Today's Betting Opportunities")
    today_data = make_request("/api/predictions/today")

    if today_data:
        c1, c2 = st.columns(2)
        c1.metric("Games Analyzed", today_data.get("total_games", 0))
        c2.metric("Bets Recommended", today_data.get("bets_recommended", 0))

        predictions = today_data.get("predictions", [])
        bets = [p for p in predictions if p["verdict"].startswith("Bet")]

        if bets:
            st.success(f"{len(bets)} betting opportunity(s) found!")
            for bet in bets:
                g = bet.get("game", {})
                matchup = f"{g.get('away_team')} @ {g.get('home_team')}"
                game_time = datetime.fromisoformat(g.get("game_date")).strftime("%b %d, %I:%M %p UTC")

                # Determine pick string from full_analysis
                spread_value = None
                if bet.get("full_analysis"):
                    spread_value = bet["full_analysis"].get("inputs", {}).get("odds", {}).get("spread")

                projected_margin = bet.get("projected_margin", 0.0)
                if spread_value is not None:
                    if projected_margin > spread_value:
                        pick_str = f"{g.get('home_team')} {spread_value:+.1f}"
                    else:
                        pick_str = f"{g.get('away_team')} {-spread_value:+.1f}"
                else:
                    pick_str = "See verdict"

                with st.expander(f"{matchup} — {game_time}", expanded=True):
                    st.subheader(pick_str)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Projected Margin", f"{bet.get('projected_margin', 0):.1f} pts")
                    col2.metric("Conservative Edge", f"{bet.get('edge_conservative', 0):.2%}")
                    col3.metric("Recommended Stake", f"{bet.get('recommended_units', 0):.2f} units")
                    st.info(f"**Verdict:** {bet['verdict']}")
        else:
            st.info("No bets recommended today (PASS on all games).")
            st.caption("This is expected 85–95% of the time in efficient markets.")
    else:
        st.warning("No prediction data available for today.")


# Performance page has moved to pages/1_Performance.py


# ==============================================================================
# BET LOG PAGE
# ==============================================================================

elif page == "📋 Bet Log":
    st.title("Bet Log")

    tab_add, tab_settle, tab_history = st.tabs(["Add Bet", "Settle Pending", "History"])

    # --------------------------------------------------------------------------
    # TAB 1: ADD BET
    # --------------------------------------------------------------------------
    with tab_add:
        st.subheader("Log a New Bet")
        st.caption("Use this to record bets you placed at a sportsbook, or manually add paper trades.")

        # Fetch recent games for the selector
        games_data = make_request("/api/games/recent", {"days_back": 7, "days_ahead": 2})
        game_options = {}
        if games_data and games_data.get("games"):
            for g in games_data["games"]:
                dt = datetime.fromisoformat(g["game_date"]).strftime("%b %d")
                label = f"{g['matchup']} ({dt})"
                game_options[label] = g["id"]

        with st.form("add_bet_form", clear_on_submit=True):
            if game_options:
                selected_label = st.selectbox("Game", list(game_options.keys()))
                game_id = game_options[selected_label]
            else:
                st.warning("No recent/upcoming games found. Enter game ID manually.")
                game_id = st.number_input("Game ID", min_value=1, step=1, value=1)

            col1, col2 = st.columns(2)
            with col1:
                pick = st.text_input(
                    "Pick",
                    placeholder='e.g. "Duke -4.5" or "Kansas +3"',
                    help="Team name + spread, exactly as you bet it.",
                )
                bet_type = st.selectbox("Bet Type", ["spread", "moneyline", "total"])
                odds_taken = st.number_input(
                    "Odds Taken (American)",
                    value=-110,
                    step=5,
                    help="e.g. -110, +105, -220",
                )

            with col2:
                bet_size_units = st.number_input(
                    "Bet Size (units)", min_value=0.1, max_value=10.0, value=1.0, step=0.5
                )
                bet_size_dollars = st.number_input(
                    "Bet Size ($)", min_value=1.0, value=10.0, step=5.0
                )
                is_paper_trade = st.checkbox("Paper Trade (simulated)", value=False)

            notes = st.text_area("Notes (optional)", max_chars=500)

            submitted = st.form_submit_button("Log Bet", type="primary")

        if submitted:
            if not pick or len(pick) < 2:
                st.error("Pick is required (minimum 2 characters).")
            elif odds_taken == 0 or (-99 < odds_taken < 100):
                st.error("Odds must be valid American odds (>= +100 or <= -100).")
            else:
                payload = {
                    "game_id": int(game_id),
                    "pick": pick,
                    "bet_type": bet_type,
                    "odds_taken": float(odds_taken),
                    "bet_size_units": float(bet_size_units),
                    "bet_size_dollars": float(bet_size_dollars),
                    "is_paper_trade": is_paper_trade,
                    "notes": notes or None,
                }
                result = make_post_request("/api/bets/log", payload)
                if result:
                    st.success(
                        f"Bet #{result['bet_id']} logged: **{result['pick']}** "
                        f"({result['bet_size_units']} units)"
                    )

    # --------------------------------------------------------------------------
    # TAB 2: SETTLE PENDING BETS
    # --------------------------------------------------------------------------
    with tab_settle:
        st.subheader("Pending Bets — Record Outcomes")
        st.caption("Update each pending bet with its result and closing line.")

        pending_data = make_request("/api/bets", {"status": "pending", "days": 60})

        if not pending_data or not pending_data.get("bets"):
            st.info("No pending bets found.")
        else:
            pending_bets = pending_data["bets"]
            st.write(f"**{len(pending_bets)} pending bet(s)**")

            for bet in pending_bets:
                ts = datetime.fromisoformat(bet["timestamp"]).strftime("%b %d") if bet.get("timestamp") else "?"
                header = f"#{bet['id']} — {bet['pick']} | {bet['matchup']} | {ts}"

                with st.expander(header):
                    col1, col2, col3 = st.columns(3)
                    model_prob_str = f"{bet['model_prob']:.1%}" if bet.get("model_prob") else "N/A"
                    col1.write(f"**Odds taken:** {bet['odds_taken']:+g}")
                    col2.write(f"**Size:** {bet['bet_size_units']} u / ${bet['bet_size_dollars']:.2f}")
                    col3.write(f"**Model prob:** {model_prob_str}")

                    with st.form(f"settle_{bet['id']}"):
                        outcome = st.radio(
                            "Result", ["Win", "Loss"],
                            horizontal=True,
                            key=f"outcome_{bet['id']}",
                        )

                        st.markdown("**Closing Line (optional — enables CLV calculation)**")
                        cl1, cl2, cl3 = st.columns(3)
                        with cl1:
                            closing_spread = st.number_input(
                                "Closing Spread",
                                value=0.0,
                                step=0.5,
                                key=f"cs_{bet['id']}",
                                help="Closing spread for your side (e.g. -6.0)",
                            )
                        with cl2:
                            closing_odds = st.number_input(
                                "Closing Odds",
                                value=-110,
                                step=5,
                                key=f"co_{bet['id']}",
                            )
                        with cl3:
                            closing_odds_other = st.number_input(
                                "Closing Odds (other side)",
                                value=-110,
                                step=5,
                                key=f"coo_{bet['id']}",
                            )
                        use_clv = st.checkbox(
                            "Include closing line data",
                            value=False,
                            key=f"use_clv_{bet['id']}",
                        )

                        settle_notes = st.text_input("Notes", key=f"notes_{bet['id']}")

                        if st.form_submit_button("Record Outcome", type="primary"):
                            payload = {"outcome": 1 if outcome == "Win" else 0}
                            if use_clv:
                                payload["closing_spread"] = float(closing_spread) if closing_spread != 0 else None
                                payload["closing_odds"] = float(closing_odds)
                                payload["closing_odds_other_side"] = float(closing_odds_other)
                            if settle_notes:
                                payload["notes"] = settle_notes

                            result = make_put_request(f"/api/bets/{bet['id']}/outcome", payload)
                            if result:
                                pl = result.get("profit_loss_dollars", 0) or 0
                                clv_grade = result.get("clv_grade") or ""
                                st.success(
                                    f"{'Win' if result['outcome'] == 1 else 'Loss'} recorded. "
                                    f"P&L: **${pl:+.2f}**"
                                    + (f" | CLV: {clv_grade}" if clv_grade else "")
                                )
                                st.rerun()

    # --------------------------------------------------------------------------
    # TAB 3: HISTORY
    # --------------------------------------------------------------------------
    with tab_history:
        st.subheader("Bet History")

        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            history_days = st.selectbox("Window", [7, 14, 30, 60, 90, 180], index=3, key="hist_days")
        with col_filter2:
            history_status = st.selectbox("Status", ["all", "settled", "pending"], key="hist_status")

        all_bets = make_request("/api/bets", {"status": history_status, "days": history_days})

        if all_bets and all_bets.get("bets"):
            bets_list = all_bets["bets"]
            df = pd.DataFrame(bets_list)

            # Format outcome column
            df["result"] = df["outcome"].map({1: "Win", 0: "Loss", None: "Pending"}).fillna("Pending")

            display_cols = [
                "id", "matchup", "game_date", "pick", "bet_type",
                "odds_taken", "bet_size_units", "bet_size_dollars",
                "result", "profit_loss_dollars", "profit_loss_units",
                "clv_points", "clv_prob", "is_paper_trade",
            ]
            # Keep only columns that exist in the dataframe
            display_cols = [c for c in display_cols if c in df.columns]

            st.write(f"**{len(df)} bet(s)**")
            st.dataframe(
                df[display_cols].rename(columns={
                    "id": "ID",
                    "matchup": "Matchup",
                    "game_date": "Game Date",
                    "pick": "Pick",
                    "bet_type": "Type",
                    "odds_taken": "Odds",
                    "bet_size_units": "Units",
                    "bet_size_dollars": "Dollars",
                    "result": "Result",
                    "profit_loss_dollars": "P&L ($)",
                    "profit_loss_units": "P&L (u)",
                    "clv_points": "CLV pts",
                    "clv_prob": "CLV prob",
                    "is_paper_trade": "Paper?",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Summary row for settled bets
            settled = df[df["result"] != "Pending"]
            if not settled.empty:
                total_pl = settled["profit_loss_dollars"].sum()
                total_risked = settled["bet_size_dollars"].sum()
                wins = (settled["result"] == "Win").sum()
                st.markdown(
                    f"**Summary:** {len(settled)} settled | "
                    f"{wins}W-{len(settled)-wins}L | "
                    f"P&L: **${total_pl:+.2f}** | "
                    f"ROI: **{total_pl/total_risked:.1%}**" if total_risked > 0 else ""
                )
        else:
            st.info("No bets found for this filter.")


# Footer
st.markdown("---")
st.caption("CBB Edge Analyzer v7.0 | Built with Streamlit")
