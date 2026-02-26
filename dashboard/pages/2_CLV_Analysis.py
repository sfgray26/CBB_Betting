"""CLV Analysis page."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard.utils import api_get, sidebar_api_key, SEVERITY_COLORS

st.set_page_config(page_title="CLV Analysis | CBB Edge", layout="wide")
sidebar_api_key()

st.title("CLV Analysis")
st.caption(
    "Closing Line Value measures whether you beat the closing market price. "
    "Sustained positive CLV is the strongest indicator of a real edge."
)

clv = api_get("/api/performance/clv-analysis")

if not clv or clv.get("bets_with_clv", 0) == 0:
    st.info(
        "No CLV data yet. CLV is calculated when closing lines are captured "
        "before game time, or supplied manually when settling a bet."
    )
    st.stop()

# --- Status banner ---
status = clv.get("status", "UNKNOWN")
rec = clv.get("recommendation", "")
if status == "HEALTHY":
    st.success(f"CLV Status: HEALTHY — {rec}")
elif status == "WARNING":
    st.warning(f"CLV Status: WARNING — {rec}")
elif status == "STOP":
    st.error(f"CLV Status: STOP — {rec}")

# --- Top metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean CLV",      f"{clv.get('mean_clv', 0):.2%}")
c2.metric("Median CLV",    f"{clv.get('median_clv', 0):.2%}")
c3.metric("Positive CLV%", f"{clv.get('positive_clv_rate', 0):.1%}")
c4.metric("Bets w/ CLV",   clv.get("bets_with_clv", 0))

st.markdown("---")

col_l, col_r = st.columns(2)

# --- Distribution histogram ---
dist = clv.get("distribution", {})
with col_l:
    st.subheader("CLV Distribution")
    labels = ["< -3%", "-3% to -1%", "-1% to +1%", "+1% to +3%", "> +3%"]
    counts = [
        dist.get("strong_negative", 0),
        dist.get("negative", 0),
        dist.get("neutral", 0),
        dist.get("positive", 0),
        dist.get("strong_positive", 0),
    ]
    colors = ["#d62728", "#ff7f0e", "#aec7e8", "#98df8a", "#2ca02c"]
    fig_dist = go.Figure(go.Bar(
        x=labels, y=counts, marker_color=colors, text=counts, textposition="outside",
    ))
    fig_dist.add_vline(x=2.0, line_dash="dot", line_color="navy",
                       annotation_text="0% CLV")
    fig_dist.update_layout(height=320, showlegend=False, xaxis_title="CLV Bucket", yaxis_title="Count")
    st.plotly_chart(fig_dist, use_container_width=True)

# --- CLV by confidence level ---
conf = clv.get("clv_by_confidence", {})
with col_r:
    st.subheader("CLV by Edge Confidence")
    if conf:
        df_conf = pd.DataFrame([
            {"confidence": k.replace("_", " ").title(), "mean_clv": v.get("mean_clv") or 0, "count": v.get("count", 0)}
            for k, v in conf.items()
        ])
        fig_conf = px.bar(
            df_conf, x="confidence", y="mean_clv", text="count",
            color="mean_clv", color_continuous_scale="RdYlGn",
            labels={"mean_clv": "Mean CLV", "confidence": "Edge Level"},
        )
        fig_conf.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_conf.update_traces(texttemplate="%{text} bets", textposition="outside")
        fig_conf.update_layout(height=320, coloraxis_showscale=False, yaxis_tickformat=".1%")
        st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# --- Top / Bottom 10 ---
col_top, col_bot = st.columns(2)

top_10 = clv.get("top_10_clv", [])
if top_10:
    with col_top:
        st.subheader("Top 10 CLV Bets")
        df_top = pd.DataFrame(top_10)
        df_top["clv_prob"] = df_top["clv_prob"].map(lambda x: f"{x:.2%}" if x else "—")
        df_top["outcome"] = df_top["outcome"].map({1: "Win", 0: "Loss"})
        st.dataframe(df_top[["bet_id", "pick", "game_date", "clv_prob", "outcome"]],
                     hide_index=True, use_container_width=True)

bot_10 = clv.get("bottom_10_clv", [])
if bot_10:
    with col_bot:
        st.subheader("Bottom 10 CLV Bets")
        df_bot = pd.DataFrame(bot_10)
        df_bot["clv_prob"] = df_bot["clv_prob"].map(lambda x: f"{x:.2%}" if x else "—")
        df_bot["outcome"] = df_bot["outcome"].map({1: "Win", 0: "Loss"})
        st.dataframe(df_bot[["bet_id", "pick", "game_date", "clv_prob", "outcome"]],
                     hide_index=True, use_container_width=True)

st.markdown("---")

# --- CLV vs. Realization Scatter Plot ---
scatter_data = clv.get("scatter_data", [])
if scatter_data and len(scatter_data) > 0:
    st.subheader("CLV vs. Realized Profit")
    st.caption(
        "This plot shows the relationship between Closing Line Value (CLV) and actual profit/loss. "
        "A positive correlation indicates that CLV is predictive of profitability."
    )

    df_scatter = pd.DataFrame(scatter_data)

    # Create scatter plot with trendline (try with OLS, fall back to no trendline if statsmodels not available)
    try:
        fig_scatter = px.scatter(
            df_scatter,
            x="clv_prob",
            y="profit_loss_units",
            color="outcome",
            color_discrete_map={"Win": "#2ca02c", "Loss": "#d62728"},
            hover_data=["pick"],
            trendline="ols",
            labels={
                "clv_prob": "CLV (Probability Edge)",
                "profit_loss_units": "Profit/Loss (Units)",
                "outcome": "Result",
            },
        )
    except Exception as e:
        # Fallback without trendline if statsmodels not installed
        st.warning("Trendline unavailable (install statsmodels for OLS regression line)")
        fig_scatter = px.scatter(
            df_scatter,
            x="clv_prob",
            y="profit_loss_units",
            color="outcome",
            color_discrete_map={"Win": "#2ca02c", "Loss": "#d62728"},
            hover_data=["pick"],
            labels={
                "clv_prob": "CLV (Probability Edge)",
                "profit_loss_units": "Profit/Loss (Units)",
                "outcome": "Result",
            },
        )

    # Add quadrant lines
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_scatter.update_layout(
        height=450,
        xaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Calculate correlation
    if len(df_scatter) > 1:
        corr = np.corrcoef(df_scatter["clv_prob"], df_scatter["profit_loss_units"])[0, 1]
        st.caption(f"📊 Correlation: **{corr:.3f}** (1.0 = perfect positive, -1.0 = perfect negative, 0.0 = no correlation)")
else:
    st.info("Scatter plot requires settled bets with both CLV and P&L data.")
