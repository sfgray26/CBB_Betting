"""Model Calibration page."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.utils import api_get, sidebar_api_key

st.set_page_config(page_title="Calibration | CBB Edge", layout="wide")
sidebar_api_key()

st.title("Model Calibration")
st.caption(
    "A well-calibrated model's 60% predictions should win ~60% of the time. "
    "Systematic over/under-confidence indicates the model needs recalibration."
)

calib = api_get("/api/performance/calibration")

if not calib or not calib.get("calibration_buckets"):
    st.info("Not enough settled bets with model_prob data for calibration analysis.")
    st.stop()

buckets = calib.get("calibration_buckets", [])
mean_err = calib.get("mean_calibration_error")
brier = calib.get("brier_score")
well_cal = calib.get("is_well_calibrated")

# --- Header metrics ---
c1, c2, c3 = st.columns(3)
c1.metric(
    "Mean Calibration Error",
    f"{mean_err:.1%}" if mean_err is not None else "N/A",
    delta="Well calibrated" if well_cal else ("Recalibration suggested" if well_cal is False else ""),
)
c2.metric("Brier Score", f"{brier:.3f}" if brier is not None else "N/A",
          help="Lower is better. Perfect = 0, coin flip ≈ 0.25")
c3.metric("Calibration Status",
          "✅ Good" if well_cal else ("⚠️ Drift" if well_cal is False else "—"))

st.markdown("---")

df = pd.DataFrame(buckets)

# --- Calibration curve ---
st.subheader("Calibration Curve")
fig = go.Figure()

# Perfect calibration diagonal
fig.add_trace(go.Scatter(
    x=[0.50, 0.82], y=[0.50, 0.82],
    mode="lines", name="Perfect calibration",
    line=dict(dash="dash", color="gray"),
))

# Model calibration
fig.add_trace(go.Scatter(
    x=df["predicted_prob"],
    y=df["actual_win_rate"],
    mode="markers+lines",
    name="Model",
    marker=dict(
        size=df["count"].clip(5, 40),
        sizemode="area",
        color=df["error"],
        colorscale="RdYlGn_r",
        showscale=True,
        colorbar=dict(title="Error"),
    ),
    hovertemplate=(
        "Bin: %{customdata[0]}<br>"
        "Predicted: %{x:.1%}<br>"
        "Actual: %{y:.1%}<br>"
        "n=%{customdata[1]}<extra></extra>"
    ),
    customdata=list(zip(df["bin"], df["count"])),
))

fig.update_layout(
    xaxis=dict(title="Predicted Win Probability", tickformat=".0%", range=[0.48, 0.84]),
    yaxis=dict(title="Actual Win Rate",           tickformat=".0%", range=[0.30, 0.95]),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# --- Data table ---
st.subheader("Bucket Detail")
df_display = df.copy()
df_display["predicted_prob"] = df_display["predicted_prob"].map("{:.1%}".format)
df_display["actual_win_rate"] = df_display["actual_win_rate"].map("{:.1%}".format)
df_display["error"] = df_display["error"].map("{:.1%}".format)
st.dataframe(
    df_display.rename(columns={
        "bin": "Bin", "predicted_prob": "Predicted", "actual_win_rate": "Actual",
        "count": "Sample (n)", "error": "Error",
    }),
    hide_index=True, use_container_width=True,
)

st.markdown("---")
st.markdown(
    "**Interpretation:**\n"
    "- **Error < 3%** — excellent calibration, no action needed\n"
    "- **Error 3–7%** — acceptable, monitor over next 50+ bets\n"
    "- **Error > 7%** — recalibration suggested: adjust `BASE_SD` or model weights\n\n"
    "**Brier score** benchmarks: 0.20 = good for binary sports outcomes, < 0.18 = excellent"
)
