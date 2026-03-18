"""
Streamlit Dashboard for CBB Edge Analyzer
Landing page — system health, quick stats, navigation guide.
Individual views live in dashboard/pages/.
"""

import sys
import os
# Ensure the project root (parent of dashboard/) is on the path so that
# "from dashboard.shared import ..." resolves correctly regardless of how
# Streamlit was launched.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), encoding="utf-8")

API_KEY = os.getenv("API_KEY_USER1", "")

from dashboard.shared import inject_custom_css
from dashboard.utils import api_get, sidebar_api_key

inject_custom_css()

# Sidebar — API key setup
with st.sidebar:
    st.title("CBB Edge")
    st.caption("v8.0 Betting Framework")

    if not API_KEY:
        key_input = st.text_input("API Key", type="password", key="api_key_input")
        if key_input:
            st.session_state["api_key"] = key_input
            st.success("Key set for this session")
    else:
        st.session_state.setdefault("api_key", API_KEY)

    st.markdown("---")
    st.caption("Use the sidebar pages to navigate.")

# Main content — landing page
st.title("CBB Edge Analyzer")

# System health check
health = api_get("/health")
if health and health.get("status") == "healthy":
    st.success("API is healthy")
else:
    st.error("Could not reach the API. Is the backend running?")

# Quick stats
perf = api_get("/api/performance/summary")
overall = perf.get("overall", perf) if perf else {}
_total = perf.get("total_bets", 0) or overall.get("total_bets", 0) if perf else 0
if perf and _total > 0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bets", _total)
    c2.metric("Win Rate", f"{overall.get('win_rate', 0):.1%}")
    roi = overall.get("roi", 0) or 0
    c3.metric("ROI", f"{roi:.1%}")
    clv = overall.get("mean_clv", 0) or 0
    c4.metric("Mean CLV", f"{clv:.2%}")
else:
    st.info("No settled bets yet. Start by running the nightly analysis.")

st.markdown("---")

# Navigation guide
st.subheader("Pages")
st.markdown("""
| Page | Description |
|------|-------------|
| **Dashboard** | Overview metrics and recent recommendations |
| **Today's Bets** | BET/CONSIDER verdicts and parlay tickets |
| **Bet Log** | Add bets, settle pending, view history |
| **Morning Briefing** | Daily snapshot before first tipoff |
| **Admin Panel** | Manual job triggers, portfolio gauges, config |
| **Performance** | ROI, win rate, cumulative P&L charts |
| **CLV Analysis** | Closing line value distribution and trends |
| **Bet History** | Filterable table with CSV export |
| **Calibration** | Model probability calibration curve |
| **Alerts** | Active alerts and system monitoring |
""")

# Footer
st.markdown("---")
st.caption("CBB Edge Analyzer v8.0")
