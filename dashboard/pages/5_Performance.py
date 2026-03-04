"""Performance Overview page."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dashboard.utils import api_get, sidebar_api_key
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Performance | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("Performance Overview")

perf = api_get("/api/performance/summary")

_total_bets = perf.get("total_bets", 0) or perf.get("overall", {}).get("total_bets", 0) if perf else 0
if not perf or _total_bets == 0:
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

st.markdown("---")

# --- Financial / Risk Metrics ---
st.subheader("Financial Risk Metrics")
fin_days = st.select_slider("Financial metrics window (days)", [30, 60, 90, 180], value=90,
                             key="fin_days_slider")
fin = api_get("/api/performance/financial-metrics", {"days": fin_days})

if not fin or fin.get("total_bets", 0) == 0:
    st.info("Not enough settled bets for financial metrics (requires at least 10 bets with P&L data).")
else:
    f1, f2, f3, f4, f5 = st.columns(5)
    sharpe  = fin.get("sharpe")
    sortino = fin.get("sortino")
    eg      = fin.get("expected_kelly_growth")
    dd_pct  = fin.get("max_drawdown_pct")
    calmar  = fin.get("calmar")

    def _fmt_ratio(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None else "—"

    f1.metric("Sharpe Ratio",    _fmt_ratio(sharpe),
              help="Annualised risk-adjusted return. > 1.0 is good, > 2.0 is excellent.")
    f2.metric("Sortino Ratio",   _fmt_ratio(sortino),
              help="Like Sharpe but penalises only downside volatility.")
    f3.metric("Exp. Kelly Growth", _fmt_ratio(eg, 4),
              help="Log-utility expected growth per bet. Positive = edge over book.")
    f4.metric("Max Drawdown",    f"{dd_pct:.1%}" if dd_pct is not None else "—",
              help="Largest peak-to-trough loss as a percentage of peak equity.")
    f5.metric("Calmar Ratio",    _fmt_ratio(calmar),
              help="Annualised return ÷ max drawdown. > 1.0 is healthy.")

    # Daily P&L bar chart
    daily_series = fin.get("daily_pl_series", {})
    if daily_series:
        st.subheader("Daily P&L")
        pl_dates  = sorted(daily_series.keys())
        pl_values = [daily_series[d] for d in pl_dates]
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pl_values]
        fig_daily = go.Figure(go.Bar(
            x=pl_dates, y=pl_values,
            marker_color=bar_colors,
            hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
        ))
        fig_daily.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_daily.update_layout(
            xaxis_title="Date", yaxis_title="Daily P&L ($)",
            height=280, showlegend=False,
        )
        st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("---")

# --- Dynamic Source Weights ---
st.subheader("Dynamic Ensemble Weights")
st.caption("Per-source weights auto-calibrated from rolling 14-day margin MAE. Updates nightly.")

weights_data = api_get("/api/performance/source-weights")
if weights_data:
    w_kenpom = weights_data.get("weight_kenpom", 0.342)
    w_bt     = weights_data.get("weight_barttorvik", 0.333)
    w_em     = weights_data.get("weight_evanmiya", 0.325)
    changed  = weights_data.get("last_changed_at")
    method   = weights_data.get("last_changed_by", "—")

    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("KenPom",     f"{w_kenpom:.3f}", help="Proportion of composite margin attributed to KenPom ratings")
    wc2.metric("BartTorvik", f"{w_bt:.3f}",     help="Proportion attributed to BartTorvik")
    wc3.metric("EvanMiya",   f"{w_em:.3f}",     help="Proportion attributed to EvanMiya")

    if changed:
        try:
            ts = datetime.fromisoformat(changed.replace("Z", "+00:00")).strftime("%b %d %H:%M UTC")
        except Exception:
            ts = changed[:16]
        st.caption(f"Last calibrated: **{ts}** (via *{method}*)")

    # Visual weight bar
    total = w_kenpom + w_bt + w_em or 1.0
    fig_w = go.Figure(go.Bar(
        x=["KenPom", "BartTorvik", "EvanMiya"],
        y=[w_kenpom / total, w_bt / total, w_em / total],
        marker_color=["#3498db", "#e67e22", "#9b59b6"],
        text=[f"{v/total:.1%}" for v in [w_kenpom, w_bt, w_em]],
        textposition="outside",
    ))
    fig_w.update_layout(
        yaxis=dict(range=[0, 0.7], title="Weight"),
        height=250,
        showlegend=False,
    )
    st.plotly_chart(fig_w, use_container_width=True)
else:
    st.info("Source weight data unavailable — run the nightly analysis to initialise.")
