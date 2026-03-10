"""
DEBUG: Raw Bets Viewer — Bypass all UI logic

This is a standalone page that directly queries the API and shows raw data.
Use this while the main Today's Bets page has issues.
"""

import streamlit as st
import requests
import os

st.set_page_config(page_title="DEBUG: Raw Bets | CBB Edge", layout="wide")

st.title("🐛 DEBUG: Raw Bets Viewer")
st.warning("This is a debug page. Use 'Today's Bets' for normal viewing.")

# Get API key from session or env
api_key = st.session_state.get("api_key") or os.getenv("API_KEY_USER1", "")

if not api_key:
    api_key = st.text_input("API Key", type="password")
    if api_key:
        st.session_state["api_key"] = api_key
        st.rerun()
else:
    # Try to get bets from debug endpoint
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load Bets (Last 24h)", type="primary"):
            try:
                # Use the debug endpoint
                response = requests.get(
                    "https://cbb-betting-production.up.railway.app/admin/debug/bets-last-24h",
                    headers={"X-API-Key": api_key},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state["debug_bets"] = data
                    st.success(f"Loaded {data.get('bet_count', 0)} bets")
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.code(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("🔄 Load from /today/all"):
            try:
                response = requests.get(
                    "https://cbb-betting-production.up.railway.app/api/predictions/today/all",
                    headers={"X-API-Key": api_key},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state["today_all"] = data
                    st.success(f"Loaded {data.get('bets_recommended', 0)} bets")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display debug endpoint data
    if "debug_bets" in st.session_state:
        data = st.session_state["debug_bets"]
        
        st.subheader("📊 Debug Endpoint Results")
        
        metrics = st.columns(4)
        metrics[0].metric("Total Predictions", data.get('total_predictions', 0))
        metrics[1].metric("BET Count", data.get('bet_count', 0))
        metrics[2].metric("Since", data.get('since', 'N/A')[:10])
        metrics[3].metric("Status", "✅ Working" if data.get('bet_count', 0) > 0 else "⚠️ No bets")
        
        bets = data.get('bets', [])
        
        if bets:
            st.subheader(f"🎯 {len(bets)} Bets Found")
            
            for bet in bets:
                with st.expander(f"{bet.get('away_team')} @ {bet.get('home_team')}"):
                    st.json(bet)
        else:
            st.info("No bets in last 24h")
    
    # Display /today/all data
    if "today_all" in st.session_state:
        data = st.session_state["today_all"]
        
        st.subheader("📊 /api/predictions/today/all Results")
        
        metrics = st.columns(3)
        metrics[0].metric("Total Games", data.get('total_games', 0))
        metrics[1].metric("Bets Recommended", data.get('bets_recommended', 0))
        metrics[2].metric("Date", str(data.get('date', 'N/A')))
        
        predictions = data.get('predictions', [])
        bets = [p for p in predictions if p.get('verdict', '').startswith('Bet')]
        
        if bets:
            st.subheader(f"🎯 {len(bets)} BET Predictions")
            
            for bet in bets[:20]:  # Limit to 20
                game = bet.get('game', {})
                with st.expander(f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}"):
                    st.json(bet)
        else:
            st.info("No BET predictions found")

st.markdown("---")
st.caption("If this page shows bets but 'Today's Bets' doesn't, the UI code hasn't deployed yet.")
