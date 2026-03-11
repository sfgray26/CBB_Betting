"""Bet History page — filterable table with CSV export, duplicate detection, and per-team breakdown."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.utils import api_get, sidebar_api_key
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Bet History | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("Bet History")

# --- Filters ---
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    days = st.selectbox("Date window", [7, 14, 30, 60, 90, 180, 365], index=4)
with col_f2:
    status_filter = st.selectbox("Status", ["all", "settled", "pending"])
with col_f3:
    paper_filter = st.selectbox("Trade type", ["all", "paper only", "real only"])

data = api_get("/api/bets", {"days": days, "status": status_filter})

if not data or not data.get("bets"):
    st.info("No bets found for this filter.")
    st.stop()

df = pd.DataFrame(data["bets"])

# Apply paper/real filter client-side
if paper_filter == "paper only":
    df = df[df["is_paper_trade"] == True]
elif paper_filter == "real only":
    df = df[df["is_paper_trade"] == False]

if df.empty:
    st.info("No bets after applying filters.")
    st.stop()

# Derived columns
df["result"] = df["outcome"].map({1: "Win", 0: "Loss", -1: "Push", None: "Pending"}).fillna("Pending")
df["game_date"] = pd.to_datetime(df["game_date"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
df["bet_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
# Flag rows where the same game_id appears more than once on the same bet_date
if "game_id" in df.columns and "bet_date" in df.columns:
    dup_key = df.groupby(["game_id", "bet_date"]).size().reset_index(name="_dup_count")
    df = df.merge(dup_key, on=["game_id", "bet_date"], how="left")
    has_duplicates = (df["_dup_count"] > 1).any()
else:
    has_duplicates = False
    df["_dup_count"] = 1

if has_duplicates:
    dup_count = (df["_dup_count"] > 1).sum()
    st.warning(
        f"**Duplicate entries detected:** {dup_count} bets share a game_id + date with another entry. "
        "These inflate bet counts and distort ROI. Use the deduplicate toggle below to hide extras, "
        "or run `/admin/debug/duplicate-bets` to see the full list."
    )

dedup_toggle = st.checkbox(
    "Deduplicate (keep only the first bet per game per day)",
    value=has_duplicates,
    help=(
        "When checked, only the earliest BetLog entry per game per day is shown. "
        "This gives the true unique-game view for accurate ROI calculation."
    ),
)

if dedup_toggle and has_duplicates:
    # Sort by id ascending so we keep the first-created entry per game/day
    df = df.sort_values("id", ascending=True)
    df = df.drop_duplicates(subset=["game_id", "bet_date"], keep="first")
    st.caption(f"After deduplication: **{len(df)} unique bets**")

# Drop helper columns before display
df = df.drop(columns=["_dup_count", "bet_date"], errors="ignore")

# --- Sort controls ---
sort_col, sort_dir = st.columns(2)
with sort_col:
    sort_by = st.selectbox("Sort by", ["game_date", "profit_loss_dollars", "clv_prob", "bet_size_dollars"])
with sort_dir:
    ascending = st.radio("Direction", ["Descending", "Ascending"], horizontal=True) == "Ascending"

df = df.sort_values(sort_by, ascending=ascending, na_position="last")

# --- Display columns ---
display_cols = [c for c in [
    "id", "game_date", "matchup", "pick", "bet_type", "odds_taken",
    "bet_size_units", "bet_size_dollars", "result",
    "profit_loss_dollars", "profit_loss_units",
    "clv_points", "clv_prob", "is_paper_trade",
] if c in df.columns]

rename_map = {
    "id": "ID", "game_date": "Date", "matchup": "Matchup", "pick": "Pick",
    "bet_type": "Type", "odds_taken": "Odds", "bet_size_units": "Units",
    "bet_size_dollars": "Risk ($)", "result": "Result",
    "profit_loss_dollars": "P&L ($)", "profit_loss_units": "P&L (u)",
    "clv_points": "CLV pts", "clv_prob": "CLV prob", "is_paper_trade": "Paper?",
}

st.write(f"**{len(df)} bet(s)**")
st.dataframe(
    df[display_cols].rename(columns=rename_map),
    use_container_width=True,
    hide_index=True,
)

# --- Summary row ---
settled = df[df["result"].isin(["Win", "Loss"])]
if not settled.empty:
    wins = (settled["result"] == "Win").sum()
    total_pl = settled["profit_loss_dollars"].sum()
    total_risked = settled["bet_size_dollars"].sum()
    roi = total_pl / total_risked if total_risked > 0 else 0
    st.markdown(
        f"**Summary (settled):** {len(settled)} bets | "
        f"{wins}W–{len(settled)-wins}L | "
        f"P&L **${total_pl:+.2f}** | ROI **{roi:.1%}**"
    )

# ---------------------------------------------------------------------------
# Per-team breakdown — anomaly detection for mapping issues
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Per-Team Performance")
st.caption(
    "Teams with consistently high or low win rates (≥3 bets, win rate ≥80% or ≤20%) are flagged. "
    "Systematically outlier results often indicate a team name mapping issue where the wrong "
    "KenPom/BartTorvik ratings are being used."
)

team_days = st.selectbox("Team breakdown window (days)", [30, 60, 90, 180, 365], index=2, key="team_days")
team_min_bets = st.number_input("Minimum bets to show team", min_value=1, max_value=10, value=2, key="team_min")

team_data = api_get("/api/performance/by-team", {"days": team_days, "min_bets": int(team_min_bets)})

if not team_data or not team_data.get("teams"):
    st.info("No per-team data available (requires settled bets).")
else:
    teams_df = pd.DataFrame(team_data["teams"])
    anomalies = team_data.get("anomalies", [])

    if anomalies:
        st.error(
            f"**{len(anomalies)} team(s) flagged as anomalies** (win rate ≥80% or ≤20% with ≥3 bets). "
            "These may indicate team name mapping errors — verify in `team_mapping.py`."
        )
        anom_df = pd.DataFrame(anomalies)[["team", "bets", "wins", "losses", "win_rate", "roi", "mean_edge"]]
        anom_df["win_rate"] = anom_df["win_rate"].map("{:.1%}".format)
        anom_df["roi"] = anom_df["roi"].map("{:.1%}".format)
        anom_df["mean_edge"] = anom_df["mean_edge"].map(lambda x: f"{x:.1%}" if x is not None else "—")
        st.dataframe(anom_df.rename(columns={
            "team": "Team", "bets": "Bets", "wins": "W", "losses": "L",
            "win_rate": "Win Rate", "roi": "ROI", "mean_edge": "Avg Edge",
        }), use_container_width=True, hide_index=True)
    else:
        st.success("No team mapping anomalies detected in this window.")

    # Full team table
    with st.expander(f"All teams ({len(teams_df)} teams, {team_data.get('total_bets', 0)} total bets)"):
        disp = teams_df[["team", "bets", "wins", "losses", "win_rate", "roi", "total_pl_dollars", "mean_edge", "anomaly_flag"]].copy()
        disp["win_rate"] = disp["win_rate"].map("{:.1%}".format)
        disp["roi"] = disp["roi"].map("{:.1%}".format)
        disp["mean_edge"] = disp["mean_edge"].map(lambda x: f"{x:.1%}" if x is not None else "—")
        disp["total_pl_dollars"] = disp["total_pl_dollars"].map("${:+.2f}".format)
        st.dataframe(disp.rename(columns={
            "team": "Team", "bets": "Bets", "wins": "W", "losses": "L",
            "win_rate": "Win Rate", "roi": "ROI",
            "total_pl_dollars": "Total P&L", "mean_edge": "Avg Edge",
            "anomaly_flag": "Flag?",
        }), use_container_width=True, hide_index=True)

    # Win rate chart — sorted so outliers stand out visually
    if len(teams_df) >= 3:
        fig = px.bar(
            teams_df.sort_values("win_rate"),
            x="win_rate",
            y="team",
            orientation="h",
            color="win_rate",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            labels={"win_rate": "Win Rate", "team": "Team"},
            title="Win Rate by Team (sorted) — outliers may indicate mapping issues",
            text=teams_df.sort_values("win_rate")["bets"].map(lambda n: f"{n}b"),
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
        fig.update_layout(height=max(300, len(teams_df) * 22), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# --- CSV export ---
st.markdown("---")
csv = df[display_cols].rename(columns=rename_map).to_csv(index=False).encode("utf-8")
st.download_button(
    label="Export to CSV",
    data=csv,
    file_name="cbb_edge_bet_history.csv",
    mime="text/csv",
)
