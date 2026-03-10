"""
Connection Test — Verify API connectivity
"""

import streamlit as st
import requests
import os
import sys

st.set_page_config(page_title="Connection Test | CBB Edge")

st.title("🔌 Connection Test")

# DIRECT check - don't rely on cached imports
api_url_from_env = os.getenv("API_URL", "NOT SET")

st.subheader("Raw Environment Check")
st.code(f"""
Raw os.getenv('API_URL'): {api_url_from_env}
""")

# Check if we can import utils and what it shows
try:
    # Force reimport by clearing cache
    if 'dashboard.utils' in sys.modules:
        del sys.modules['dashboard.utils']
    
    from dashboard.utils import _API_URL, _API_KEY
    
    st.subheader("Imported from dashboard.utils")
    st.code(f"""
_API_URL = {_API_URL}
_API_KEY set: {'Yes' if _API_KEY else 'No'}
""")
    
    # Check if they match
    if api_url_from_env != "NOT SET" and api_url_from_env != _API_URL:
        st.warning("⚠️ ENV VAR differs from imported _API_URL!")
    
except Exception as e:
    st.error(f"Import error: {e}")
    _API_URL = None
    _API_KEY = ""

# Use the imported URL or fallback
TEST_URL = api_url_from_env if api_url_from_env != "NOT SET" else (_API_URL if _API_URL else "https://cbb-betting-production.up.railway.app")

st.subheader("Testing URL")
st.info(f"Will test: `{TEST_URL}`")

# Test endpoints
col1, col2 = st.columns(2)

with col1:
    if st.button("Test /health"):
        try:
            response = requests.get(f"{TEST_URL}/health", timeout=10)
            st.write(f"Status: {response.status_code}")
            if response.status_code == 200:
                st.success("✅ Backend reachable!")
            else:
                st.error(f"❌ Status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

with col2:
    if st.button("Test /api/predictions/today"):
        try:
            headers = {"X-API-Key": _API_KEY if _API_KEY else ""}
            response = requests.get(
                f"{TEST_URL}/api/predictions/today",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ Got {data.get('total_games', 0)} games")
            elif response.status_code == 401:
                st.error("❌ 401 Unauthorized")
            else:
                st.error(f"❌ Status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Manual override section
st.markdown("---")
st.subheader("Manual Override")
manual_url = st.text_input(
    "Test custom URL",
    value="https://cbb-betting-production.up.railway.app"
)

if st.button("Test custom URL"):
    try:
        r = requests.get(f"{manual_url}/health", timeout=10)
        if r.status_code == 200:
            st.success(f"✅ Works: {manual_url}")
        else:
            st.error(f"❌ Status {r.status_code}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

