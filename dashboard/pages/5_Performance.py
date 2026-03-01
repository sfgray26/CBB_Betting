"""Performance Overview page."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dashboard.utils import api_get, sidebar_api_key

st.set_page_config(page_title="Performance | CBB Edge", layout="wide")
sidebar_api_key()

st.title("Performance Overview")

perf = api_get("/api/performance/summary")

if not perf or perf.get("total_bets", 0) == 0:
    st.info("No settled bets yet.")
    st.stop()

overall = perf.get("overall", perf)  # works for both old and new shapes

# --- Key metrics ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Bets",  overall.get("total_bets", 0))
c2.metric("Win Rate",    f"{overall.get('win_rate', 0):.1%}")
c3.metric("ROI",         f"{overall.get('roi', 0):.2%}")
c4.metric("Mean CLV",    f"{overall.get('mean_clv', 0) or 0:.2%}")
c5.metric("Drawdown",    f"{overall.get('current_drawdown', 0):.1%}")

status = overall.get("status", "UNKNOWN")
if status == "HEALTHY":
    st.success("System Status: HEALTHY — positive CLV sustained")
elif status == "WARNING":
    st.warning("System Status: WARNING — CLV near zero, monitor closely")
elif status == "STOP":
    st.error("System Status: STOP BETTING — CLV negative")
else:
    st.info("Status: insufficient data")

st.markdown("---")

# --- Timeline window selector ---
days = st.select_slider("History window (days)", [7, 14, 30, 60, 90, 180], value=30)
timeline_data = api_get("/api/performance/timeline", {"days": days})

if timeline_data and timeline_data.get("timeline"):
    df = pd.DataFrame(timeline_data["timeline"])
    cum_profit = timeline_data.get("cumulative_profit", [])

    # Cumulative P&L
    st.subheader("Cumulative P&L")
    fig_pl = go.Figure()
    fig_pl.add_trace(go.Scatter(
        x=list(range(1, len(cum_profit) + 1)),
        y=cum_profit,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,180,0,0.1)" if (cum_profit[-1] if cum_profit else 0) >= 0 else "rgba(220,0,0,0.1)",
        line=dict(color="green" if (cum_profit[-1] if cum_profit else 0) >= 0 else "red"),
    ))
    fig_pl.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_pl.update_layout(
        xaxis_title="Bet Number", yaxis_title="Cumulative P&L ($)", height=320,
    )
    st.plotly_chart(fig_pl, use_container_width=True)

    # Daily Capital Deployed chart
    st.subheader("Daily Capital Deployed")
    if "capital_deployed_units" in df.columns:
        max_daily_exposure_pct = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "15.0"))

        fig_capital = go.Figure()
        fig_capital.add_trace(go.Bar(
            x=df["date"],
            y=df["capital_deployed_units"],
            name="Capital Deployed",
            marker_color="steelblue",
            text=df["capital_deployed_units"].round(1),
            textposition="outside",
        ))
        fig_capital.add_hline(
            y=max_daily_exposure_pct,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max Daily Exposure ({max_daily_exposure_pct:.0f}u)",
            annotation_position="right",
        )
        fig_capital.update_layout(
            xaxis_title="Date",
            yaxis_title="Units Deployed",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_capital, use_container_width=True)
    else:
        st.info("Capital deployment data not available. Update backend to latest version.")

    col_l, col_r = st.columns(2)

    # Win rate by bet type
    by_type = perf.get("by_bet_type", {})
    if by_type:
        with col_l:
            st.subheader("Win Rate by Bet Type")
            df_type = pd.DataFrame(
                [{"type": t, **v} for t, v in by_type.items()]
            )
            fig_type = px.bar(
                df_type, x="type", y="win_rate", color="roi",
                color_continuous_scale="RdYlGn",
                labels={"win_rate": "Win Rate", "type": "Bet Type"},
                text_auto=".1%",
            )
            fig_type.update_layout(height=280, coloraxis_showscale=False)
            st.plotly_chart(fig_type, use_container_width=True)

    # Edge bucket breakdown
    by_edge = perf.get("by_edge_bucket", {})
    if by_edge:
        with col_r:
            st.subheader("Performance by Edge Bucket")
            df_edge = pd.DataFrame(
                [{"edge": k, **v} for k, v in by_edge.items()]
            )
            fig_edge = px.bar(
                df_edge, x="edge", y="roi", color="win_rate",
                color_continuous_scale="RdYlGn",
                labels={"roi": "ROI", "edge": "Conservative Edge"},
                text_auto=".1%",
            )
            fig_edge.update_layout(height=280, coloraxis_showscale=False)
            st.plotly_chart(fig_edge, use_container_width=True)

st.markdown("---")

# --- Rolling windows ---
rolling = perf.get("rolling_windows", {})
if rolling:
    st.subheader("Rolling Windows")
    cols = st.columns(len(rolling))
    for col, (label, data) in zip(cols, rolling.items()):
        col.metric(
            label.replace("_", " ").title(),
            f"Win Rate {data.get('win_rate', 0):.1%}",
            delta=f"CLV {data.get('mean_clv', 0) or 0:.2%}",
        )

st.markdown("---")

# --- Model Accuracy (margin prediction + calibration) ---
st.subheader("Model Accuracy")
acc_days = st.select_slider("Accuracy window (days)", [14, 30, 60, 90, 180], value=90)
acc = api_get("/api/performance/model-accuracy", {"days": acc_days})

if not acc or acc.get("count", 0) == 0:
    st.info(
        "No resolved predictions yet. "
        "Outcome data populates automatically within 2 hours of each game finishing."
    )
else:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Predictions resolved", acc.get("count", 0))
    a2.metric("Margin MAE",  f"{acc.get('mean_mae', 0) or 0:.2f} pts")
    a3.metric("Signed error", f"{acc.get('mean_signed_error', 0) or 0:+.2f} pts",
              help="Positive = model over-predicts home margin on average")
    a4.metric("Brier score", f"{acc.get('brier_score', 0) or 0:.4f}",
              help="Lower is better. Perfect calibration = 0, random = 0.25")

    col_l, col_r = st.columns(2)

    # MAE by verdict
    mae_verdict = acc.get("mae_by_verdict", {})
    if mae_verdict:
        with col_l:
            st.markdown("**Margin MAE by verdict**")
            for verdict, mae in mae_verdict.items():
                st.write(f"- {verdict}: **{mae:.2f} pts**")

    # MAE by rating source
    mae_source = acc.get("mae_by_source", {})
    if any(v is not None for v in mae_source.values()):
        with col_r:
            st.markdown("**Margin MAE by rating source**")
            for src, mae in mae_source.items():
                if mae is not None:
                    st.write(f"- {src.title()}: **{mae:.2f} pts**")

    # Probability calibration table
    calib = acc.get("calibration", [])
    if calib:
        st.markdown("**Probability calibration (predicted win prob vs actual win rate)**")
        calib_df = pd.DataFrame(calib)
        calib_df["gap"] = (calib_df["actual_win_rate"] - calib_df["predicted_prob"]).map(
            lambda x: f"{x:+.1%}"
        )
        calib_df.columns = [c.replace("_", " ").title() for c in calib_df.columns]
        st.dataframe(calib_df, use_container_width=True, hide_index=True)
