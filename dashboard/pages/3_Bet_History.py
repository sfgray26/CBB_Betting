"""Bet History page — filterable table with CSV export."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import streamlit as st
from dashboard.utils import api_get, sidebar_api_key

st.set_page_config(page_title="Bet History | CBB Edge", layout="wide")
sidebar_api_key()

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
df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

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

# --- CSV export ---
st.markdown("---")
csv = df[display_cols].rename(columns=rename_map).to_csv(index=False).encode("utf-8")
st.download_button(
    label="Export to CSV",
    data=csv,
    file_name="cbb_edge_bet_history.csv",
    mime="text/csv",
)
