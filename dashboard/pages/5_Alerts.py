"""Alerts & Monitoring page."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from datetime import datetime
from dashboard.utils import api_get, api_post, sidebar_api_key, SEVERITY_COLORS

st.set_page_config(page_title="Alerts | CBB Edge", layout="wide")
sidebar_api_key()

ADMIN_KEY = os.getenv("API_KEY_USER1", "")

st.title("Alerts & System Monitoring")

# --- Live alerts ---
alerts_data = api_get("/api/performance/alerts")

live_alerts = alerts_data.get("live_alerts", []) if alerts_data else []
system_status = alerts_data.get("status", "UNKNOWN") if alerts_data else "UNKNOWN"

if system_status == "CRITICAL":
    st.error("System Status: CRITICAL — immediate action required")
elif system_status == "WARNING":
    st.warning("System Status: WARNING — review recommended")
elif system_status == "OK":
    st.success("System Status: OK — no issues detected")
else:
    st.info("System Status: UNKNOWN — insufficient data")

st.markdown("---")

if live_alerts:
    st.subheader(f"Active Alerts ({len(live_alerts)})")
    for alert in sorted(live_alerts, key=lambda a: ["INFO", "WARNING", "CRITICAL"].index(a.get("severity", "INFO"))):
        icon = SEVERITY_COLORS.get(alert.get("severity", "INFO"), "⚪")
        severity = alert.get("severity", "INFO")

        with st.expander(f"{icon} **{severity}** — {alert.get('alert_type', '').replace('_', ' ')}", expanded=(severity != "INFO")):
            st.write(alert.get("message", ""))
            if alert.get("recommendation"):
                st.info(f"**Action:** {alert['recommendation']}")
            cols = st.columns(2)
            cols[0].metric("Current Value", f"{alert.get('current_value', '—')}")
            cols[1].metric("Threshold",     f"{alert.get('threshold', '—')}")
else:
    st.success("No active alerts — all systems nominal.")

st.markdown("---")

# --- Historical alerts ---
st.subheader("Alert History")
show_ack = st.checkbox("Show acknowledged alerts", value=False)
hist_data = api_get("/api/performance/alerts", {"include_acknowledged": str(show_ack).lower()})
hist_alerts = hist_data.get("alerts", []) if hist_data else []

if hist_alerts:
    for a in hist_alerts:
        icon = SEVERITY_COLORS.get(a.get("severity"), "⚪")
        ack_tag = " ✓" if a.get("acknowledged") else ""
        ts = a.get("created_at", "")[:16].replace("T", " ") if a.get("created_at") else "?"
        with st.expander(f"{icon} [{ts}] {a.get('alert_type', '').replace('_', ' ')}{ack_tag}"):
            st.write(a.get("message", ""))
            st.caption(f"Value: {a.get('current_value')} | Threshold: {a.get('threshold')}")
            if not a.get("acknowledged") and ADMIN_KEY:
                if st.button(f"Acknowledge alert #{a['id']}", key=f"ack_{a['id']}"):
                    import requests
                    r = requests.post(
                        f"{os.getenv('API_URL', 'http://localhost:8000')}/admin/alerts/{a['id']}/acknowledge",
                        headers={"X-API-Key": ADMIN_KEY},
                        timeout=10,
                    )
                    if r.ok:
                        st.success("Acknowledged")
                        st.rerun()
else:
    st.info("No alert history found.")

st.markdown("---")

# --- System health ---
st.subheader("System Health Checks")

health = api_get("/health")
c1, c2, c3 = st.columns(3)

if health:
    db_status = health.get("database", "unknown")
    sched_status = health.get("scheduler", "unknown")
    c1.metric("Database",  "✅ Connected" if db_status == "connected" else f"❌ {db_status}")
    c2.metric("Scheduler", "✅ Running"   if sched_status == "running" else f"❌ {sched_status}")
    c3.metric("API",       "✅ Healthy"   if health.get("status") == "healthy" else "⚠️ Degraded")
else:
    st.error("Could not reach the API. Is the backend running?")

# Scheduler jobs
sched = api_get("/admin/scheduler/status") if ADMIN_KEY else None
if sched and sched.get("jobs"):
    st.subheader("Scheduled Jobs")
    for job in sched["jobs"]:
        next_run = job.get("next_run", "—")
        if next_run and next_run != "—":
            next_run = next_run[:19].replace("T", " ")
        st.write(f"- **{job['name']}** — next run: `{next_run}`")
