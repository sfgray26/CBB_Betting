"""Dashboard Overview — key metrics, system status, recent recommendations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import streamlit as st
from datetime import datetime
from dashboard.utils import api_get, sidebar_api_key
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Dashboard | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("CBB Edge Analyzer")

perf = api_get("/api/performance/summary")

# Stats live under "overall"; fall back to flat dict for backward compat
overall = perf.get("overall", perf) if perf else {}

col1, col2, col3, col4 = st.columns(4)
if perf and overall.get("total_bets", 0) > 0:
    with col1:
        st.metric("Total Bets", overall["total_bets"])
    with col2:
        st.metric("Win Rate", f"{overall.get('win_rate', 0):.1%}")
    with col3:
        roi = overall.get("roi", 0) or 0
        st.metric("ROI", f"{roi:.1%}", delta="positive" if roi > 0 else "negative")
    with col4:
        clv = overall.get("mean_clv", 0) or 0
        st.metric("Mean CLV", f"{clv:.2%}")

    st.markdown("---")
    status = overall.get("status", "UNKNOWN")
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
bets_data = api_get("/api/predictions/bets", {"days": 7})
if bets_data and bets_data.get("bets"):
    st.dataframe(pd.DataFrame(bets_data["bets"]), use_container_width=True)
else:
    st.info("No bets recommended in the last 7 days.")
