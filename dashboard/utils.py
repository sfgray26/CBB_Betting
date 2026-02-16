"""Shared utilities for all dashboard pages."""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_API_URL = os.getenv("API_URL", "http://localhost:8000")
_API_KEY = os.getenv("API_KEY_USER1", "")


def _key() -> str:
    return st.session_state.get("api_key", _API_KEY)


def _headers() -> dict:
    return {"X-API-Key": _key()}


def api_get(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{_API_URL}{endpoint}", headers=_headers(), params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(
            f"{_API_URL}{endpoint}",
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        st.error(f"API {exc.response.status_code}: {detail}")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def api_put(endpoint: str, payload: dict):
    try:
        r = requests.put(
            f"{_API_URL}{endpoint}",
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        st.error(f"API {exc.response.status_code}: {detail}")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def sidebar_api_key() -> None:
    """Show API key input in sidebar if the key is not yet set."""
    if not _key():
        with st.sidebar:
            key_input = st.text_input("API Key", type="password", key="api_key_sidebar")
            if key_input:
                st.session_state["api_key"] = key_input
                st.rerun()


SEVERITY_COLORS = {
    "CRITICAL": "🔴",
    "WARNING":  "🟡",
    "INFO":     "🔵",
}
