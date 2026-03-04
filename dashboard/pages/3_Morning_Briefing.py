"""Morning Briefing — daily snapshot of predictions, portfolio, alerts, scheduler."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from dashboard.utils import api_get, sidebar_api_key
from dashboard.shared import inject_custom_css, SEVERITY_COLORS

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
