"""
MINIMAL Today's Bets — Stripped down version for troubleshooting

This is a minimal version that should work even if the main page has issues.
"""

import streamlit as st
import requests
import os
from datetime import datetime

st.set_page_config(page_title="MINIMAL: Today's Bets | CBB Edge", layout="wide")

st.title("⚡ MINIMAL: Today's Bets")
st.caption("Stripped-down version for troubleshooting")

# API key handling
api_key = st.session_state.get("api_key") or os.getenv("API_KEY_USER1", "")

if not api_key:
    api_key = st.text_input("Enter API Key", type="password")
    if api_key:
        st.session_state["api_key"] = api_key
        st.rerun()
    st.stop()

# Let user choose endpoint
endpoint_choice = st.radio(
    "Select Data Source:",
    ["Last 24h (Debug)", "Today/All API", "Standard Today API"],
    index=0
)

if st.button("Load Data", type="primary"):
    with st.spinner("Loading..."):
        try:
            # Determine endpoint
            if endpoint_choice == "Last 24h (Debug)":
                url = "https://cbb-betting-production.up.railway.app/admin/debug/bets-last-24h"
            elif endpoint_choice == "Today/All API":
                url = "https://cbb-betting-production.up.railway.app/api/predictions/today/all"
            else:
                url = "https://cbb-betting-production.up.railway.app/api/predictions/today"
            
            response = requests.get(
                url,
                headers={"X-API-Key": api_key},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state["minimal_data"] = data
                st.session_state["endpoint_used"] = endpoint_choice
                st.success(f"✅ Loaded from {endpoint_choice}")
            else:
                st.error(f"❌ API Error {response.status_code}")
                st.code(response.text[:500])
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display data
if "minimal_data" in st.session_state:
    data = st.session_state["minimal_data"]
    endpoint = st.session_state.get("endpoint_used", "Unknown")
    
    st.subheader(f"Results from: {endpoint}")
    
    # Extract bets based on endpoint
    if "bets" in data:  # Debug endpoint
        bets = data["bets"]
        total = data.get("total_predictions", 0)
    elif "predictions" in data:  # Today endpoints
        predictions = data["predictions"]
        total = len(predictions)
        bets = [p for p in predictions if p.get("verdict", "").startswith("Bet")]
    else:
        bets = []
        total = 0
    
    # Show metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", total)
    c2.metric("BET Count", len(bets))
    c3.metric("API Endpoint", endpoint.split()[0])
    
    # Show bets
    if bets:
        st.subheader(f"🎯 {len(bets)} Bets")
        
        for i, bet in enumerate(bets[:30], 1):  # Show first 30
            if "home_team" in bet:  # Debug format
                title = f"{i}. {bet.get('away_team')} @ {bet.get('home_team')}"
                edge = bet.get('edge', 0)
                units = bet.get('units', 0)
            else:  # Standard prediction format
                game = bet.get('game', {})
                title = f"{i}. {game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}"
                edge = bet.get('edge_conservative', 0) or 0
                units = bet.get('recommended_units', 0) or 0
            
            with st.expander(f"{title} — Edge: {edge:.1%}, Units: {units:.2f}"):
                st.json(bet)
    else:
        st.info("No bets found")
        st.json(data)  # Show raw data for debugging

st.markdown("---")
st.caption(f"Current UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} | API Key: {'✅ Set' if api_key else '❌ Missing'}")
