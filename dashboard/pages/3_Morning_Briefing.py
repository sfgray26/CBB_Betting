"""Morning Briefing — daily snapshot of predictions, portfolio, alerts, scheduler."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from dashboard.utils import api_get, sidebar_api_key
from dashboard.shared import inject_custom_css, SEVERITY_COLORS

from backend.services.scout import generate_morning_briefing_narrative

st.set_page_config(page_title="Morning Briefing | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("Morning Briefing")
st.caption("Quick daily snapshot — check this before the first tipoff.")

# --- Today's predictions summary ---
st.subheader("Today's Slate")
today_data = api_get("/api/predictions/today")

if today_data:
    predictions = today_data.get("predictions", [])
    bets = [p for p in predictions if p["verdict"].startswith("Bet")]
    considers = [p for p in predictions if p["verdict"].upper().startswith("CONSIDER")]
    passes = len(predictions) - len(bets) - len(considers)

    # Narrative Briefing
    top_bet = bets[0] if bets else None
    top_bet_info = None
    if top_bet:
        g = top_bet.get("game", {})
        fa = top_bet.get("full_analysis", {})
        inputs = fa.get("inputs", {})
        odds_data = inputs.get("odds", {})
        
        home = g.get("home_team") or odds_data.get("home_team") or "Home"
        away = g.get("away_team") or odds_data.get("away_team") or "Away"
        edge = top_bet.get("edge_conservative", 0)
        top_bet_info = f"{away} @ {home} (Edge {edge:.1%})"
    
    narrative = generate_morning_briefing_narrative(len(bets), len(considers), top_bet_info)
    st.info(narrative)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games Analyzed", today_data.get("total_games", 0))
    c2.metric("BET", len(bets))
    c3.metric("CONSIDER", len(considers))
    c4.metric("PASS", passes)

    if bets:
        st.success(f"{len(bets)} actionable bet(s) today — see Today's Bets page for details.")
    elif considers:
        st.info(f"{len(considers)} marginal edge(s) — monitoring for line movement.")
    else:
        st.info("All games PASS today. No action needed.")

    # --- V9 Risk Intelligence ---
    if bets:
        st.markdown("---")
        st.subheader("V9 Risk Intelligence")
        st.caption(
            "SNR measures source agreement (KenPom / BartTorvik / EvanMiya). "
            "Integrity is the real-time OpenClaw sanity check."
        )

        _any_flag = False
        for bet in bets:
            fa    = bet.get("full_analysis", {})
            calcs = fa.get("calculations", {})
            g     = bet.get("game", {})
            fa_inputs = fa.get("inputs", {})
            odds_data = fa_inputs.get("odds", {})
            home = g.get("home_team") or odds_data.get("home_team") or "Home"
            away = g.get("away_team") or odds_data.get("away_team") or "Away"

            snr_val    = calcs.get("snr")
            int_verdict = calcs.get("integrity_verdict")
            int_scalar  = calcs.get("integrity_kelly_scalar", 1.0)
            snr_scalar  = calcs.get("snr_kelly_scalar", 1.0)

            # SNR badge
            if snr_val is None:
                snr_badge = "— SNR"
            elif snr_val >= 0.75:
                snr_badge = f"✅ SNR {snr_val:.0%}"
            elif snr_val >= 0.50:
                snr_badge = f"⚠️ SNR {snr_val:.0%}"
                _any_flag = True
            else:
                snr_badge = f"🔴 SNR {snr_val:.0%}"
                _any_flag = True

            # Integrity badge
            _iv_upper = (int_verdict or "").upper()
            if "ABORT" in _iv_upper or "RED FLAG" in _iv_upper:
                int_badge = f"🛑 ABORT"
                _any_flag = True
            elif "VOLATILE" in _iv_upper:
                int_badge = f"🔴 VOLATILE"
                _any_flag = True
            elif "CAUTION" in _iv_upper:
                int_badge = f"⚠️ CAUTION"
                _any_flag = True
            elif int_verdict:
                int_badge = "✅ CONFIRMED"
            else:
                int_badge = "— Not run"

            net_v9 = snr_scalar * int_scalar
            net_badge = f"Kelly {net_v9:.2f}×"

            col_m, col_s, col_i, col_k = st.columns([3, 2, 3, 2])
            col_m.markdown(f"**{away} @ {home}**")
            col_s.markdown(snr_badge)
            col_i.markdown(int_badge)
            col_k.markdown(net_badge)

        if _any_flag:
            st.warning(
                "One or more bets have elevated risk flags. "
                "Review the full V9 panel on Today's Bets before placing action."
            )
else:
    st.warning("No prediction data available. Has the nightly analysis run?")

st.markdown("---")

# --- Portfolio exposure ---
st.subheader("Portfolio Status")
portfolio = api_get("/admin/portfolio/status")

if portfolio:
    p1, p2, p3 = st.columns(3)
    p1.metric("Current Exposure", f"{portfolio.get('current_exposure_pct', 0):.1f}%")
    p2.metric("Drawdown", f"{portfolio.get('current_drawdown_pct', 0):.1f}%")
    p3.metric("Open Positions", portfolio.get("open_positions", 0))

    if portfolio.get("circuit_breaker_active"):
        st.error("Circuit breaker is ACTIVE — all new bets paused until drawdown recovers.")
else:
    st.info("Portfolio data unavailable (admin key required).")

st.markdown("---")

# --- Active alerts ---
st.subheader("Active Alerts")
alerts_data = api_get("/api/performance/alerts")

if alerts_data:
    live_alerts = alerts_data.get("live_alerts", [])
    if live_alerts:
        for alert in live_alerts:
            icon = SEVERITY_COLORS.get(alert.get("severity", "INFO"), "")
            st.write(f"{icon} **{alert.get('severity')}** — {alert.get('message', '')}")
    else:
        st.success("No active alerts — all systems nominal.")
else:
    st.info("Alert data unavailable.")

st.markdown("---")

# --- Next scheduled job ---
st.subheader("Scheduler")
sched = api_get("/admin/scheduler/status")

if sched and sched.get("jobs"):
    for job in sched["jobs"]:
        next_run = job.get("next_run", "—")
        if next_run and next_run != "—":
            next_run = next_run[:19].replace("T", " ")
        st.write(f"- **{job['name']}** — next run: `{next_run}`")
else:
    st.info("Scheduler data unavailable (admin key required).")
