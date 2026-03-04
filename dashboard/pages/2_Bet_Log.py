"""Bet Log — Add Bet, Settle Pending (with closing line auto-fill), History."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import streamlit as st
from datetime import datetime
from dashboard.utils import api_get, api_post, api_put, sidebar_api_key
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Bet Log | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("Bet Log")

tab_add, tab_settle, tab_history = st.tabs(["Add Bet", "Settle Pending", "History"])

# --------------------------------------------------------------------------
# TAB 1: ADD BET
# --------------------------------------------------------------------------
with tab_add:
    st.subheader("Log a New Bet")
    st.caption("Use this to record bets you placed at a sportsbook, or manually add paper trades.")

    games_data = api_get("/api/games/recent", {"days_back": 7, "days_ahead": 2})
    game_options = {}
    if games_data and games_data.get("games"):
        for g in games_data["games"]:
            try:
                dt = datetime.fromisoformat(g.get("game_date") or "").strftime("%b %d")
            except (ValueError, TypeError):
                dt = "?"
            label = f"{g['matchup']} ({dt})"
            game_options[label] = g["id"]

    if game_options:
        selected_label = st.selectbox(
            "Select Game",
            list(game_options.keys()),
            key="bet_game_selector",
            help="Model recommendation will auto-populate when you select a game."
        )
        game_id = game_options[selected_label]

        prediction_data = api_get(f"/api/predictions/game/{game_id}")
        bet_details = prediction_data.get("bet_details", {}) if prediction_data else {}

        if prediction_data and bet_details.get("has_bet"):
            st.success(f"Model Recommendation: {prediction_data.get('verdict', 'N/A')}")
        elif prediction_data:
            st.info(f"Model Verdict: {prediction_data.get('verdict', 'PASS')} — You can still log a manual bet.")
        else:
            st.warning("No prediction found for this game.")

        default_pick = bet_details.get("pick") or ""
        default_type = bet_details.get("bet_type") or "spread"
        default_odds = bet_details.get("odds") or -110
        default_units = bet_details.get("units") or 1.0
    else:
        st.warning("No recent/upcoming games found. Enter game ID manually.")
        game_id = st.number_input("Game ID", min_value=1, step=1, value=1)
        default_pick = ""
        default_type = "spread"
        default_odds = -110
        default_units = 1.0

    if default_type not in ["spread", "moneyline", "total"]:
        default_type = "spread"

    with st.form("add_bet_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            pick = st.text_input(
                "Pick",
                value=default_pick,
                placeholder='e.g. "Duke -4.5" or "Kansas +3"',
                help="Team name + spread, exactly as you bet it. Auto-populated from model.",
            )
            bet_type = st.selectbox(
                "Bet Type",
                ["spread", "moneyline", "total"],
                index=["spread", "moneyline", "total"].index(default_type)
            )
            odds_taken = st.number_input(
                "Odds Taken (American)",
                value=int(default_odds),
                step=5,
                help="e.g. -110, +105, -220. Auto-populated from model.",
            )

        with col2:
            bet_size_units = st.number_input(
                "Bet Size (units)",
                min_value=0.1,
                max_value=10.0,
                value=float(default_units),
                step=0.5,
                help="Auto-populated from model's Kelly recommendation."
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
            result = api_post("/api/bets/log", payload)
            if result:
                st.success(
                    f"Bet #{result['bet_id']} logged: **{result['pick']}** "
                    f"({result['bet_size_units']} units)"
                )

# --------------------------------------------------------------------------
# TAB 2: SETTLE PENDING BETS (with closing line auto-fill)
# --------------------------------------------------------------------------
with tab_settle:
    st.subheader("Pending Bets — Record Outcomes")
    st.caption("Update each pending bet with its result and closing line.")

    pending_data = api_get("/api/bets", {"status": "pending", "days": 60})

    if not pending_data or not pending_data.get("bets"):
        st.info("No pending bets found.")
    else:
        pending_bets = pending_data["bets"]
        st.write(f"**{len(pending_bets)} pending bet(s)**")

        for bet in pending_bets:
            try:
                ts = datetime.fromisoformat(bet["timestamp"]).strftime("%b %d") if bet.get("timestamp") else "?"
            except (ValueError, TypeError):
                ts = "?"
            header = f"#{bet['id']} — {bet['pick']} | {bet['matchup']} | {ts}"

            with st.expander(header):
                col1, col2, col3 = st.columns(3)
                model_prob_str = f"{bet['model_prob']:.1%}" if bet.get("model_prob") else "N/A"
                col1.write(f"**Odds taken:** {bet['odds_taken']:+g}")
                col2.write(f"**Size:** {bet['bet_size_units']} u / ${bet['bet_size_dollars']:.2f}")
                col3.write(f"**Model prob:** {model_prob_str}")

                # Auto-fetch closing line for this game
                cl_data = api_get(f"/api/closing-lines/{bet['game_id']}")
                cl_spread_default = 0.0
                cl_odds_default = -110
                cl_odds_other_default = -110
                cl_found = False

                if cl_data and "spread" in cl_data:
                    cl_found = True
                    cl_spread_default = cl_data.get("spread") or 0.0
                    cl_odds_default = cl_data.get("spread_odds") or -110
                    # Use moneyline_away as "other side" proxy if available
                    cl_odds_other_default = cl_data.get("moneyline_away") or -110
                    captured = cl_data.get("captured_at", "")[:16].replace("T", " ") if cl_data.get("captured_at") else ""
                    st.success(f"Closing line found (captured {captured})")
                else:
                    st.caption("No closing line captured for this game.")

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
                            value=float(cl_spread_default),
                            step=0.5,
                            key=f"cs_{bet['id']}",
                            help="Closing spread for your side (e.g. -6.0)",
                        )
                    with cl2:
                        closing_odds = st.number_input(
                            "Closing Odds",
                            value=int(cl_odds_default),
                            step=5,
                            key=f"co_{bet['id']}",
                        )
                    with cl3:
                        closing_odds_other = st.number_input(
                            "Closing Odds (other side)",
                            value=int(cl_odds_other_default),
                            step=5,
                            key=f"coo_{bet['id']}",
                        )
                    use_clv = st.checkbox(
                        "Include closing line data",
                        value=cl_found,
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

                        result = api_put(f"/api/bets/{bet['id']}/outcome", payload)
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

    all_bets = api_get("/api/bets", {"status": history_status, "days": history_days})

    if all_bets and all_bets.get("bets"):
        bets_list = all_bets["bets"]
        df = pd.DataFrame(bets_list)

        df["result"] = df["outcome"].map({1: "Win", 0: "Loss", None: "Pending"}).fillna("Pending")

        display_cols = [
            "id", "matchup", "game_date", "pick", "bet_type",
            "odds_taken", "bet_size_units", "bet_size_dollars",
            "result", "profit_loss_dollars", "profit_loss_units",
            "clv_points", "clv_prob", "is_paper_trade",
        ]
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

        settled = df[df["result"] != "Pending"]
        if not settled.empty:
            total_pl = settled["profit_loss_dollars"].sum()
            total_risked = settled["bet_size_dollars"].sum()
            wins = (settled["result"] == "Win").sum()
            if total_risked > 0:
                st.markdown(
                    f"**Summary:** {len(settled)} settled | "
                    f"{wins}W-{len(settled)-wins}L | "
                    f"P&L: **${total_pl:+.2f}** | "
                    f"ROI: **{total_pl/total_risked:.1%}**"
                )
    else:
        st.info("No bets found for this filter.")
