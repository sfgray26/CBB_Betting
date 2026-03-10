"""
Connection Test — Verify API connectivity
"""

import streamlit as st
import requests
import os

st.set_page_config(page_title="Connection Test | CBB Edge")

st.title("🔌 Connection Test")

# Show what URL we're using
from dashboard.utils import _API_URL, _API_KEY

st.subheader("Configuration")
st.code(f"""
API_URL from env: {os.getenv('API_URL', 'NOT SET')}
_API_URL default: {_API_URL}
API_KEY set: {'Yes' if _API_KEY else 'No'}
""")

# Test direct connection
st.subheader("Direct API Test")

if st.button("Test /health endpoint"):
    try:
        response = requests.get(f"{_API_URL}/health", timeout=10)
        st.write(f"Status: {response.status_code}")
        st.write(f"Response: {response.text}")
        if response.status_code == 200:
            st.success("✅ Backend is reachable!")
        else:
            st.error(f"❌ Got status {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

if st.button("Test /api/predictions/today"):
    try:
        headers = {"X-API-Key": _API_KEY}
        response = requests.get(
            f"{_API_URL}/api/predictions/today",
            headers=headers,
            timeout=10
        )
        st.write(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Got {data.get('total_games', 0)} games")
        elif response.status_code == 401:
            st.error("❌ 401 Unauthorized — API key issue")
        else:
            st.error(f"❌ Got status {response.status_code}")
            st.code(response.text[:500])
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Manual override
st.subheader("Manual Override")
st.warning("If the above tests fail, try entering the API URL manually:")

manual_url = st.text_input(
    "API URL",
    value="https://cbb-betting-production.up.railway.app",
    placeholder="https://..."
)

if st.button("Test with manual URL"):
    try:
        response = requests.get(f"{manual_url}/health", timeout=10)
        if response.status_code == 200:
            st.success(f"✅ Manual URL works: {manual_url}")
            st.info("Set this as API_URL environment variable in Streamlit Cloud")
        else:
            st.error(f"❌ Status {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("Debug info: This page shows what URL the dashboard is trying to use")
