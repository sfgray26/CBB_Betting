"""Shared CSS and UI helpers for all dashboard pages."""

import streamlit as st


SEVERITY_COLORS = {
    "CRITICAL": "🔴",
    "WARNING":  "🟡",
    "INFO":     "🔵",
}

_CUSTOM_CSS = """
<style>
/* ---- Typography ---- */
.big-metric   { font-size: 24px; font-weight: bold; }
.mono         { font-family: monospace; font-size: 13px; }

/* ---- Colour tokens ---- */
.positive     { color: #2ecc71; font-weight: bold; }
.negative     { color: #e74c3c; font-weight: bold; }
.warning      { color: #f39c12; font-weight: bold; }
.neutral      { color: #95a5a6; }

/* ---- Verdict badges ---- */
.bet-badge    { background:#1a7a3c; color:#fff; padding:3px 10px; border-radius:6px; font-weight:bold; font-size:13px; }
.cons-badge   { background:#b8860b; color:#fff; padding:3px 10px; border-radius:6px; font-weight:bold; font-size:13px; }
.pass-badge   { background:#3a3a3a; color:#ccc; padding:3px 10px; border-radius:6px; font-size:13px; }

/* ---- Status pills ---- */
.pill-green   { background:#1a4a2a; color:#2ecc71; padding:2px 8px; border-radius:12px; font-size:12px; }
.pill-yellow  { background:#3a2a00; color:#f39c12; padding:2px 8px; border-radius:12px; font-size:12px; }
.pill-red     { background:#4a1a1a; color:#e74c3c; padding:2px 8px; border-radius:12px; font-size:12px; }
.pill-grey    { background:#2a2a2a; color:#888; padding:2px 8px; border-radius:12px; font-size:12px; }

/* ---- Metric cards ---- */
.metric-card  {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
    border-left: 4px solid #3498db;
}
.metric-card.green { border-left-color: #2ecc71; }
.metric-card.red   { border-left-color: #e74c3c; }
.metric-card.gold  { border-left-color: #f39c12; }

/* ---- Tables ---- */
.stDataFrame thead th { background: #1e1e2e !important; }
</style>
"""


def inject_custom_css() -> None:
    """Inject shared CSS styles into the current page."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Return a coloured status badge string for display."""
    mapping = {
        "HEALTHY": "✅ HEALTHY",
        "OK": "✅ OK",
        "WARNING": "⚠️ WARNING",
        "CRITICAL": "🔴 CRITICAL",
        "STOP": "🛑 STOP",
        "UNKNOWN": "❓ UNKNOWN",
    }
    return mapping.get(status.upper(), f"❓ {status}")
