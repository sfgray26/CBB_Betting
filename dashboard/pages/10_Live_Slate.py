"""Live Slate — real-time model vs. market comparison with auto-refresh."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import streamlit as st
from datetime import datetime, timezone
from dashboard.utils import api_get, sidebar_api_key
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Live Slate | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

# ---------------------------------------------------------------------------
# Custom CSS for edge colour-coding
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.bet-badge   { background:#1a7a3c; color:#fff; padding:3px 10px; border-radius:6px; font-weight:bold; font-size:13px; }
.cons-badge  { background:#b8860b; color:#fff; padding:3px 10px; border-radius:6px; font-weight:bold; font-size:13px; }
.pass-badge  { background:#444;    color:#ccc; padding:3px 10px; border-radius:6px; font-size:13px; }
.edge-pos    { color:#2ecc71; font-weight:bold; }
.edge-warn   { color:#f39c12; font-weight:bold; }
.edge-neg    { color:#888; }
.monitor-bar { background:#1e1e2e; border-radius:8px; padding:8px 16px; margin-bottom:12px; }
.mono        { font-family: monospace; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page header + refresh controls
# ---------------------------------------------------------------------------
hdr_col, ctrl_col = st.columns([5, 1])
with hdr_col:
    st.title("Live Slate")
    st.caption("Model predictions vs. current market lines — updates every 60 seconds.")

with ctrl_col:
    auto_refresh = st.toggle("Auto-refresh", value=True, key="live_auto_refresh")
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
st.caption(f"Last loaded: **{now_str}**")

# ---------------------------------------------------------------------------
# Odds monitor status bar
# ---------------------------------------------------------------------------
monitor_data = api_get("/admin/odds-monitor/status")
if monitor_data:
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    is_running = monitor_data.get("is_running", False)
    last_poll  = monitor_data.get("last_poll_at")
    games_tracked = monitor_data.get("games_tracked", 0)
    quota_remaining = monitor_data.get("quota_remaining")
    quota_low = monitor_data.get("quota_is_low", False)

    m_col1.metric("Monitor", "LIVE" if is_running else "PAUSED",
                  delta="active" if is_running else "inactive")

    if last_poll:
        try:
            lp = datetime.fromisoformat(last_poll.replace("Z", "+00:00"))
            ago_s = int((datetime.now(timezone.utc) - lp).total_seconds())
            poll_str = f"{ago_s // 60}m {ago_s % 60}s ago"
        except Exception:
            poll_str = last_poll[:16]
    else:
        poll_str = "—"
    m_col2.metric("Last Poll", poll_str)
    m_col3.metric("Games Tracked", games_tracked)

    if quota_remaining is not None:
        q_delta = "LOW" if quota_low else "OK"
        m_col4.metric("API Quota", quota_remaining, delta=q_delta,
                      delta_color="inverse" if quota_low else "normal")
    else:
        m_col4.metric("API Quota", "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# Today's predictions
# ---------------------------------------------------------------------------
today_data = api_get("/api/predictions/today")

if not today_data:
    st.warning("No data from backend — is the API running?")
    st.stop()

predictions = today_data.get("predictions", [])
total_games = today_data.get("total_games", 0)
bets_rec    = today_data.get("bets_recommended", 0)

summary_c1, summary_c2, summary_c3, summary_c4 = st.columns(4)
summary_c1.metric("Games on Slate", total_games)
summary_c2.metric("Bets Recommended", bets_rec)
summary_c3.metric("CONSIDER", sum(1 for p in predictions if p["verdict"].upper().startswith("CONSIDER")))
summary_c4.metric("PASS", sum(1 for p in predictions if p["verdict"].upper().startswith("PASS")))

if not predictions:
    st.info("No predictions available yet — run the nightly analysis first.")
    st.stop()

# ---------------------------------------------------------------------------
# Split into tiers
# ---------------------------------------------------------------------------
bets      = sorted([p for p in predictions if p["verdict"].startswith("Bet")],
                   key=lambda x: x.get("edge_conservative") or 0, reverse=True)
considers = sorted([p for p in predictions if p["verdict"].upper().startswith("CONSIDER")],
                   key=lambda x: x.get("edge_conservative") or 0, reverse=True)
passes    = [p for p in predictions if p["verdict"].upper().startswith("PASS")]


def _team_names(pred: dict) -> tuple[str, str]:
    g  = pred.get("game", {})
    fa = pred.get("full_analysis", {})
    od = fa.get("inputs", {}).get("odds", {})
    home = g.get("home_team") or od.get("home_team") or "Home"
    away = g.get("away_team") or od.get("away_team") or "Away"
    return home, away


def _game_time_str(pred: dict) -> str:
    g = pred.get("game", {})
    try:
        return datetime.fromisoformat(g.get("game_date") or "").strftime("%I:%M %p")
    except (ValueError, TypeError):
        return "TBD"


def _spread_str(spread_val, team: str) -> str:
    if spread_val is None:
        return "—"
    return f"{team} {spread_val:+.1f}"


# ---------------------------------------------------------------------------
# BET cards
# ---------------------------------------------------------------------------
if bets:
    st.subheader(f"BET ({len(bets)} game{'s' if len(bets) != 1 else ''})")
    for p in bets:
        home, away = _team_names(p)
        fa   = p.get("full_analysis", {})
        calcs = fa.get("inputs", {}).get("odds", {})
        model_calcs = fa.get("calculations", {})
        spread_val  = fa.get("inputs", {}).get("odds", {}).get("spread")
        sharp_spread = fa.get("inputs", {}).get("odds", {}).get("sharp_consensus_spread")
        edge  = p.get("edge_conservative", 0) or 0
        margin = p.get("projected_margin", 0) or 0
        units  = p.get("recommended_units", 0) or 0
        bet_side = model_calcs.get("bet_side", "home")
        game_time = _game_time_str(p)

        pick_str = _spread_str(spread_val, home if bet_side == "home" else away)
        if bet_side == "away" and spread_val is not None:
            pick_str = _spread_str(-spread_val, away)

        with st.container(border=True):
            b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([3, 2, 2, 2, 2])
            with b_col1:
                st.markdown(f'<span class="bet-badge">BET</span>', unsafe_allow_html=True)
                st.markdown(f"**{away} @ {home}**")
                st.caption(game_time)
            b_col2.metric("Pick", pick_str)
            b_col3.metric("Proj Margin", f"{margin:+.1f}")
            b_col4.metric("Edge", f"{edge:.1%}")
            b_col5.metric("Stake", f"{units:.2f}u")

            if sharp_spread is not None:
                st.caption(f"Sharp consensus: **{sharp_spread:+.1f}** | Model: **{margin:+.1f}** pts")

# ---------------------------------------------------------------------------
# CONSIDER cards
# ---------------------------------------------------------------------------
if considers:
    st.markdown("---")
    st.subheader(f"CONSIDER ({len(considers)})")
    st.caption("Marginal edge — watch for line movement before tipoff.")
    for p in considers:
        home, away = _team_names(p)
        fa = p.get("full_analysis", {})
        sharp_spread = fa.get("inputs", {}).get("odds", {}).get("sharp_consensus_spread")
        edge  = p.get("edge_conservative", 0) or 0
        margin = p.get("projected_margin", 0) or 0
        game_time = _game_time_str(p)

        with st.container(border=True):
            c_col1, c_col2, c_col3, c_col4 = st.columns([3, 2, 2, 2])
            with c_col1:
                st.markdown('<span class="cons-badge">CONSIDER</span>', unsafe_allow_html=True)
                st.markdown(f"**{away} @ {home}**")
                st.caption(game_time)
            c_col2.metric("Proj Margin", f"{margin:+.1f}")
            c_col3.metric("Edge", f"{edge:.1%}")
            c_col4.metric("Verdict", p.get("verdict", "CONSIDER")[:20])

# ---------------------------------------------------------------------------
# Full slate table (all games including PASS)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Full Slate — Model vs. Market")

rows = []
for p in predictions:
    home, away = _team_names(p)
    fa    = p.get("full_analysis", {})
    odds  = fa.get("inputs", {}).get("odds", {})
    calcs = fa.get("calculations", {})
    spread_val   = odds.get("spread")
    sharp_spread = odds.get("sharp_consensus_spread")
    margin = p.get("projected_margin") or 0
    edge   = p.get("edge_conservative") or 0
    verdict = p.get("verdict", "PASS")
    units   = p.get("recommended_units") or 0
    bet_side = calcs.get("bet_side", "home")
    game_time = _game_time_str(p)

    tier = "BET" if verdict.startswith("Bet") else ("CONSIDER" if verdict.upper().startswith("CONSIDER") else "PASS")

    spread_str = f"{spread_val:+.1f}" if spread_val is not None else "—"
    sharp_str  = f"{sharp_spread:+.1f}" if sharp_spread is not None else "—"
    model_str  = f"{margin:+.1f}"
    diff_str   = f"{margin - (sharp_spread or 0):+.1f}" if sharp_spread is not None else "—"
    edge_str   = f"{edge:.1%}" if edge else "—"
    side_str   = f"{home} side" if bet_side == "home" else f"{away} side"

    rows.append({
        "Time":         game_time,
        "Matchup":      f"{away} @ {home}",
        "Mkt Spread":   spread_str,
        "Sharp":        sharp_str,
        "Model":        model_str,
        "Divergence":   diff_str,
        "Edge":         edge_str,
        "Stake (u)":    f"{units:.2f}" if units else "—",
        "Side":         side_str if tier in ("BET", "CONSIDER") else "—",
        "Tier":         tier,
    })

if rows:
    import pandas as pd
    df = pd.DataFrame(rows)

    def _colour_tier(val: str) -> str:
        if val == "BET":
            return "background-color: #1a3a2a; color: #2ecc71; font-weight: bold;"
        if val == "CONSIDER":
            return "background-color: #3a2a00; color: #f39c12; font-weight: bold;"
        return "color: #888;"

    styled = (
        df.style
          .applymap(_colour_tier, subset=["Tier"])
          .set_properties(**{"text-align": "center"}, subset=df.columns[2:])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("No game data to display.")

# ---------------------------------------------------------------------------
# Auto-refresh loop (must be at the end so the page renders first)
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(60)
    st.rerun()
