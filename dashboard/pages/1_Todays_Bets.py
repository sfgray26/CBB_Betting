"""Today's Betting Opportunities — BET cards, CONSIDER section, parlay tickets."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from datetime import datetime
from dashboard.utils import api_get, api_post, sidebar_api_key
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Today's Bets | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("Today's Betting Opportunities")
today_data = api_get("/api/predictions/today")

if today_data:
    c1, c2 = st.columns(2)
    c1.metric("Games Analyzed", today_data.get("total_games", 0))
    c2.metric("Bets Recommended", today_data.get("bets_recommended", 0))

    predictions = today_data.get("predictions", [])
    bets = [p for p in predictions if p["verdict"].startswith("Bet")]
    bets.sort(key=lambda b: b.get("edge_conservative") or 0.0, reverse=True)

    if bets:
        st.success(f"{len(bets)} betting opportunity(s) found!")
        for bet in bets:
            g = bet.get("game", {})
            fa = bet.get("full_analysis", {})
            inputs = fa.get("inputs", {})
            odds_data = inputs.get("odds", {})
            home = g.get("home_team") or odds_data.get("home_team") or "Home"
            away = g.get("away_team") or odds_data.get("away_team") or "Away"
            matchup = f"{away} @ {home}"
            try:
                game_time = datetime.fromisoformat(g.get("game_date") or "").strftime("%b %d, %I:%M %p UTC")
            except (ValueError, TypeError):
                game_time = "TBD"

            spread_value = inputs.get("odds", {}).get("spread")
            # Model odds for the recommended side
            calcs = fa.get("calculations", {})
            # Use bet_side from the model (same logic as the paper trade pick)
            bet_side = calcs.get("bet_side", "home")
            if spread_value is not None:
                if bet_side == "home":
                    pick_str = f"{home} {spread_value:+.1f}"
                else:
                    pick_str = f"{away} {-spread_value:+.1f}"
            else:
                pick_str = "See verdict"
            default_odds = int(calcs.get("bet_odds", -110) or -110)
            rec_units = bet.get("recommended_units", 0.0) or 0.0
            game_id = g.get("id") or bet.get("game_id")
            pred_id = bet.get("id")
            log_key = f"log_{game_id}"

            with st.expander(f"{matchup} — {game_time}", expanded=True):
                st.subheader(pick_str)
                col1, col2, col3 = st.columns(3)
                col1.metric("Projected Margin", f"{bet.get('projected_margin', 0):.1f} pts")
                col2.metric("Conservative Edge", f"{bet.get('edge_conservative', 0):.2%}")
                col3.metric("Recommended Stake", f"{rec_units:.2f} units")
                st.info(f"**Verdict:** {bet['verdict']}")

                # ---- "I placed this bet" button ----
                if st.session_state.get(f"logged_{game_id}"):
                    st.success("Bet logged!")
                elif st.session_state.get(log_key):
                    # Inline mini-form
                    with st.form(key=f"form_{game_id}", border=False):
                        fc1, fc2, fc3 = st.columns(3)
                        stake_dollars = fc1.number_input(
                            "Stake ($)", min_value=1.0, value=round(rec_units * 10, 2), step=5.0,
                            key=f"stake_{game_id}",
                        )
                        odds_taken = fc2.number_input(
                            "Odds", value=default_odds, step=5, key=f"odds_{game_id}",
                        )
                        submitted = fc3.form_submit_button("Confirm", type="primary", use_container_width=True)

                    if submitted:
                        payload = {
                            "game_id": game_id,
                            "pick": pick_str,
                            "bet_type": "spread",
                            "odds_taken": float(odds_taken),
                            "bet_size_units": round(rec_units, 2),
                            "bet_size_dollars": float(stake_dollars),
                            "is_paper_trade": False,
                            "notes": f"Logged from Today's Bets | pred_id={pred_id}",
                        }
                        result = api_post("/api/bets/log", payload)
                        if result:
                            st.session_state[f"logged_{game_id}"] = True
                            st.session_state.pop(log_key, None)
                            st.rerun()
                        else:
                            st.error("Failed to log bet — check API connection.")
                else:
                    if st.button("I placed this bet", key=f"btn_{game_id}"):
                        st.session_state[log_key] = True
                        st.rerun()
    else:
        st.info("No bets recommended today (PASS on all games).")
        st.caption("This is expected 85-95% of the time in efficient markets.")

    # CONSIDER verdicts
    considers = [p for p in predictions if p["verdict"].upper().startswith("CONSIDER")]
    considers.sort(key=lambda b: b.get("edge_conservative") or 0.0, reverse=True)

    if considers:
        st.markdown("---")
        st.subheader("CONSIDER — Marginal Edges (monitoring)")
        st.caption(
            "These games have a detectable edge below the MIN_BET_EDGE threshold. "
            "Watch for line movement toward the model's side before tipoff."
        )
        for c in considers:
            g = c.get("game", {})
            fa = c.get("full_analysis", {})
            inputs = fa.get("inputs", {})
            odds_data = inputs.get("odds", {})
            home = g.get("home_team") or odds_data.get("home_team") or "Home"
            away = g.get("away_team") or odds_data.get("away_team") or "Away"
            matchup = f"{away} @ {home}"
            edge = c.get("edge_conservative", 0)
            margin = c.get("projected_margin", 0)
            try:
                game_time = datetime.fromisoformat(g.get("game_date") or "").strftime("%I:%M %p")
            except (ValueError, TypeError):
                game_time = "TBD"

            with st.expander(f"{matchup} | Edge {edge:.1%} | {game_time}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Projected Margin", f"{margin:.1f} pts")
                col2.metric("Conservative Edge", f"{edge:.2%}")
                col3.metric("Verdict", c["verdict"])
else:
    st.warning("No prediction data available for today.")

# Optimal Parlays
st.markdown("---")
st.subheader("Optimal Parlays")
st.caption(
    "Cross-game parlays built from today's +EV straight bets. "
    "Sizing respects the remaining daily portfolio budget after straight bets."
)

parlay_data = api_get("/api/predictions/parlays")

if parlay_data is None:
    st.warning("Could not reach the parlay endpoint — is the API running?")
else:
    capacity_remaining = parlay_data.get("remaining_capacity_dollars")
    if capacity_remaining is not None:
        st.caption(
            f"Portfolio capacity remaining today: **${capacity_remaining:.2f}** "
            f"(of ${parlay_data.get('max_daily_dollars', 0):.2f} max daily)"
        )

    parlays = parlay_data.get("parlays", [])
    exhausted_msg = parlay_data.get("message", "")

    if not parlays:
        if "exhausted" in exhausted_msg.lower():
            st.info("Portfolio capacity exhausted — no parlay sizing available today.")
        else:
            st.info("No qualifying parlay combinations found today.")
            st.caption(
                "Parlays require 2+ straight bets each with edge > 1%. "
                "Check back after the nightly analysis runs."
            )
    else:
        st.success(f"{len(parlays)} parlay ticket(s) recommended today.")
        for i, parlay in enumerate(parlays, start=1):
            leg_summary   = parlay.get("leg_summary", "—")
            american_odds = parlay.get("parlay_american_odds", 0)
            joint_prob    = parlay.get("joint_prob", 0.0)
            edge          = parlay.get("edge", 0.0)
            rec_units     = parlay.get("recommended_units", 0.0)
            num_legs      = parlay.get("num_legs", "?")
            ev            = parlay.get("expected_value", 0.0)

            odds_str = f"+{american_odds:.0f}" if american_odds >= 0 else f"{american_odds:.0f}"
            header   = f"Parlay {i} — {num_legs}-Leg @ {odds_str}"

            with st.expander(header, expanded=(i == 1)):
                st.markdown(f"**Legs:** {leg_summary}")
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                pcol1.metric("Joint Prob",      f"{joint_prob:.1%}")
                pcol2.metric("Edge (EV/unit)",  f"{edge:.3f}")
                pcol3.metric("Expected Value",  f"{ev:.3f} u")
                pcol4.metric("Rec. Stake",      f"{rec_units:.2f} u")
