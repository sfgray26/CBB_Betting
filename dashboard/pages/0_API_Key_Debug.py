"""
API Key Debug — Check if key is being stored correctly
"""

import streamlit as st
import os

st.set_page_config(page_title="API Key Debug | CBB Edge")

st.title("🔑 API Key Debug")

# Check all the places the API key might be
st.subheader("API Key Sources")

# 1. Environment variable
env_key = os.getenv("API_KEY_USER1", "")
st.write(f"1. Environment variable `API_KEY_USER1`: {'✅ Set (' + env_key[:10] + '...)' if env_key else '❌ Not set'}")

# 2. Session state
session_key = st.session_state.get("api_key", "")
st.write(f"2. Session state `api_key`: {'✅ Set (' + session_key[:10] + '...)' if session_key else '❌ Not set'}")

# 3. What utils.py sees
from dashboard.utils import _API_KEY, _key
st.write(f"3. utils._API_KEY (from env): {'✅ Set (' + _API_KEY[:10] + '...)' if _API_KEY else '❌ Not set'}")
st.write(f"4. utils._key() (session > env): {'✅ Returns (' + _key()[:10] + '...)' if _key() else '❌ Empty'}")

st.markdown("---")

# Test API call with current key
st.subheader("Test API Call")

if st.button("Test /health with current key"):
    from dashboard.utils import api_get
    result = api_get("/health")
    if result:
        st.success(f"✅ Success: {result}")
    else:
        st.error("❌ Failed - check error above")

st.markdown("---")

# Manual key entry for testing
st.subheader("Manual Key Entry (Test)")
manual_key = st.text_input("Enter API key to test:", type="password")

if st.button("Set session key manually"):
    st.session_state["api_key"] = manual_key
    st.success("Key set! Refresh this page to see if it persists.")
    st.rerun()

if st.button("Clear session key"):
    if "api_key" in st.session_state:
        del st.session_state["api_key"]
    st.success("Session key cleared!")
    st.rerun()

st.markdown("---")
st.caption("Debug page to check API key handling")
