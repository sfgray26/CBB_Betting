"""Today's Betting Opportunities — BET cards, CONSIDER section, parlay tickets."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from datetime import datetime, date, timedelta
from dashboard.utils import api_get, api_post, sidebar_api_key
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Today's Bets | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("Today's Betting Opportunities")

# Add toggle to show all recent bets vs only upcoming
col1, col2 = st.columns([3, 1])
with col2:
    show_recent = st.toggle("Show last 24 hours (not just upcoming)", value=True, 
                           help="Show all bets from last 24h including games that may have started")

# Show current time for clarity
st.caption(f"Current UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} | Data updates every 30 min")

# Use appropriate endpoint based on toggle
if show_recent:
    # Use the /all endpoint and filter client-side for last 24h
    today_data = api_get("/api/predictions/today/all")
else:
    today_data = api_get("/api/predictions/today")

if today_data:
    predictions = today_data.get("predictions", [])
    
    # Debug info
    st.caption(f"Raw predictions from API: {len(predictions)} | Toggle 'show_recent': {show_recent}")

    # Last-resort UI dedup guard: API should already deduplicate, but protect
    # against stale cache or test data serving duplicate game_ids.
    _seen_gids: dict = {}
    for _p in predictions:
        _gid = _p.get("game_id") or (_p.get("game") or {}).get("id")
        if _gid is None:
            continue
        if _gid not in _seen_gids or (
            (_p.get("edge_conservative") or 0)
            > (_seen_gids[_gid].get("edge_conservative") or 0)
        ):
            _seen_gids[_gid] = _p
    predictions = list(_seen_gids.values())
    
    # Filter to last 24 hours if toggle is enabled
    if show_recent:
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        filtered_predictions = []
        for p in predictions:
            try:
                game_date_str = p.get("game", {}).get("game_date")
                if game_date_str:
                    # Handle both ISO format with and without timezone
                    game_date_str = game_date_str.replace('Z', '+00:00')
                    game_dt = datetime.fromisoformat(game_date_str)
                    # Convert to naive UTC for comparison
                    if game_dt.tzinfo:
                        game_dt = game_dt.replace(tzinfo=None)
                    if game_dt >= twenty_four_hours_ago:
                        filtered_predictions.append(p)
            except Exception as e:
                # If date parsing fails, include the prediction anyway
                filtered_predictions.append(p)
        predictions = filtered_predictions
        st.caption(f"After 24h filter: {len(predictions)} games")

    bets = [p for p in predictions if p.get("verdict", "").startswith("Bet")]
    bets.sort(key=lambda b: b.get("edge_conservative") or 0.0, reverse=True)
    
    # Show metrics based on filtered data
    c1, c2 = st.columns(2)
    c1.metric("Games Analyzed", len(predictions))
    c2.metric("Bets Recommended", len(bets))

    if bets:
        st.success(f"{len(bets)} betting opportunity(s) found!")

        # ---- High-Confidence Alpha Signals ----
        alpha_bets = []
        for _b in bets:
            _c = _b.get("full_analysis", {}).get("calculations", {})
            _snr = _c.get("snr") or 0.0
            _iv  = (_c.get("integrity_verdict") or "").upper()
            if _snr >= 0.9 and "CONFIRMED" in _iv:
                alpha_bets.append(_b)

        if alpha_bets:
            st.markdown("---")
            st.markdown("### ⭐ High-Confidence Alpha Signals")
            st.caption(
                "These bets pass every V9 filter: SNR ≥ 90% (near-perfect source agreement) "
                "and Integrity = CONFIRMED (real-time news check passed). "
                "Full Kelly sizing applies — no additional discount."
            )
            for _ab in alpha_bets:
                _g   = _ab.get("game", {})
                _fa  = _ab.get("full_analysis", {})
                _inp = _fa.get("inputs", {})
                _od  = _inp.get("odds", {})
                _h   = _g.get("home_team") or _od.get("home_team") or "Home"
                _a   = _g.get("away_team") or _od.get("away_team") or "Away"
                _c   = _fa.get("calculations", {})
                _snr = _c.get("snr", 0.0)
                _net = (_c.get("snr_kelly_scalar", 1.0)) * (_c.get("integrity_kelly_scalar", 1.0))
                _eu  = _ab.get("recommended_units", 0.0) or 0.0
                _edge = _ab.get("edge_conservative", 0.0) or 0.0
                st.success(
                    f"⭐ **{_a} @ {_h}** — "
                    f"Edge {_edge:.1%} · {_eu:.2f}u · "
                    f"SNR {_snr:.0%} · Net Kelly {_net:.2f}×"
                )
            st.markdown("---")

        for bet in bets:
            g = bet.get("game", {})
            fa = bet.get("full_analysis", {})
            inputs = fa.get("inputs", {})
            odds_data = inputs.get("odds", {})
            home = g.get("home_team") or odds_data.get("home_team") or "Home"
            away = g.get("away_team") or odds_data.get("away_team") or "Away"
            matchup = f"{away} @ {home}"
            
            # Check if game has already started
            game_started = False
            try:
                game_dt = datetime.fromisoformat(g.get("game_date") or "")
                game_started = game_dt < datetime.utcnow()
                game_time = game_dt.strftime("%b %d, %I:%M %p UTC")
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
            log_key = f"log_{pred_id}"

            # V9 confidence fields
            snr_val       = calcs.get("snr")
            int_verdict   = calcs.get("integrity_verdict")
            int_scalar    = calcs.get("integrity_kelly_scalar", 1.0)
            snr_scalar_v  = calcs.get("snr_kelly_scalar", 1.0)

            # Determine integrity badge
            _iv_upper = (int_verdict or "").upper()
            if "ABORT" in _iv_upper or "RED FLAG" in _iv_upper:
                _int_icon, _int_color = "🛑", "red"
            elif "VOLATILE" in _iv_upper:
                _int_icon, _int_color = "🔴", "red"
            elif "CAUTION" in _iv_upper:
                _int_icon, _int_color = "⚠️", "orange"
            elif int_verdict:
                _int_icon, _int_color = "✅", "green"
            else:
                _int_icon, _int_color = "—", "grey"

            _expander_label = f"{matchup} — {game_time}"
            if game_started:
                _expander_label += "  ⏱️ STARTED"
            elif _int_color in ("red", "orange"):
                _expander_label += f"  {_int_icon}"

            with st.expander(_expander_label, expanded=True):
                st.subheader(pick_str)
                col1, col2, col3 = st.columns(3)
                col1.metric("Projected Margin", f"{bet.get('projected_margin') or 0:.1f} pts")
                col2.metric("Conservative Edge", f"{bet.get('edge_conservative') or 0:.2%}")
                col3.metric("Recommended Stake", f"{rec_units:.2f} units")
                st.info(f"**Verdict:** {bet['verdict']}")

                # ---- V9 Confidence Panel ----
                if snr_val is not None or int_verdict:
                    st.markdown("**V9 Confidence Assessment**")
                    v_col1, v_col2, v_col3 = st.columns(3)

                    if snr_val is not None:
                        if snr_val >= 0.75:
                            _snr_icon = "✅"
                        elif snr_val >= 0.50:
                            _snr_icon = "⚠️"
                        else:
                            _snr_icon = "🔴"
                        v_col1.metric(
                            f"{_snr_icon} Source SNR",
                            f"{snr_val:.0%}",
                            delta=f"Kelly {snr_scalar_v:.2f}×",
                            delta_color="off",
                            help="Agreement across KenPom / BartTorvik / EvanMiya. Low = sources disagree → smaller bet.",
                        )

                    if int_verdict:
                        v_col2.metric(
                            f"{_int_icon} Integrity",
                            int_verdict[:24],
                            delta=f"Kelly {int_scalar:.2f}×",
                            delta_color="off",
                            help="OpenClaw Integrity Officer verdict based on real-time DuckDuckGo search.",
                        )
                    else:
                        v_col2.metric("— Integrity", "Not run", help="Sanity check only runs for BET-tier pre-scores.")

                    net_v9 = snr_scalar_v * int_scalar
                    v_col3.metric(
                        "Net V9 Kelly",
                        f"{net_v9:.2f}×",
                        delta="of base size",
                        delta_color="off",
                        help="Combined SNR × Integrity multiplier applied to Kelly fraction.",
                    )

                # ---- "I placed this bet" button ----
                if st.session_state.get(f"logged_{pred_id}"):
                    st.success("Bet logged!")
                elif st.session_state.get(log_key):
                    # Inline mini-form
                    with st.form(key=f"form_{pred_id}", border=False):
                        fc1, fc2, fc3 = st.columns(3)
                        stake_dollars = fc1.number_input(
                            "Stake ($)", min_value=1.0, value=round(rec_units * 10, 2), step=5.0,
                            key=f"stake_{pred_id}",
                        )
                        odds_taken = fc2.number_input(
                            "Odds", value=default_odds, step=5, key=f"odds_{pred_id}",
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
                            st.session_state[f"logged_{pred_id}"] = True
                            st.session_state.pop(log_key, None)
                            st.rerun()
                        else:
                            st.error("Failed to log bet — check API connection.")
                else:
                    if st.button("I placed this bet", key=f"btn_{pred_id}"):
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
            edge = c.get("edge_conservative") or 0
            margin = c.get("projected_margin") or 0
            pred_id = c.get("id")
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
