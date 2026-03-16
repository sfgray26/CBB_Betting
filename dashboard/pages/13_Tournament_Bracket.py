"""
NCAA Tournament Bracket Projection — Streamlit page.

Calls GET /api/tournament/bracket-projection and renders:
  1. Champion & Final Four projections
  2. Upset alerts
  3. Per-region bracket breakdown
  4. Full advancement probability table

Run the dashboard with:
    streamlit run dashboard/app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Tournament Bracket",
    page_icon="🏀",
    layout="wide",
)
st.title("🏀 NCAA Tournament Bracket Projection")
st.caption(
    "Monte Carlo simulation: 10,000 brackets with historically-calibrated upset probabilities"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY_USER1", "dev-key-insecure")

N_SIMS_OPTIONS = {
    "1,000 (fast)": 1000,
    "5,000": 5000,
    "10,000 (default)": 10000,
    "25,000 (slow)": 25000,
}

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Simulation Settings")
    n_sims_label = st.selectbox("Simulations", list(N_SIMS_OPTIONS.keys()), index=2)
    n_sims = N_SIMS_OPTIONS[n_sims_label]
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Running bracket simulation...")
def fetch_projection(n_sims_count: int) -> dict:
    """Call the bracket-projection endpoint and return parsed JSON."""
    url = f"{API_BASE}/api/tournament/bracket-projection"
    headers = {"X-API-Key": API_KEY}
    params = {"n_sims": n_sims_count}
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


if "bracket_result" not in st.session_state or run_btn:
    try:
        st.session_state["bracket_result"] = fetch_projection(n_sims)
        st.session_state["bracket_n_sims"] = n_sims
    except requests.HTTPError as exc:
        st.warning(
            f"Bracket projection endpoint returned HTTP {exc.response.status_code}. "
            "Is the backend running and has a tournament bracket been loaded?"
        )
        st.stop()
    except requests.ConnectionError:
        st.warning(
            f"Cannot connect to the backend at {API_BASE}. "
            "Start the API server with: `uvicorn backend.main:app --reload`"
        )
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error fetching projection: {exc}")
        st.stop()

result: dict = st.session_state["bracket_result"]
adv_probs: dict = result.get("advancement_probs", {})
upset_alerts: list = result.get("upset_alerts", [])
projected_champion: str = result.get("projected_champion", "N/A")
projected_f4: list = result.get("projected_final_four", [])
teams_resolved: int = result.get("teams_resolved", len(adv_probs))
n_sims_used: int = result.get("n_sims", n_sims)

st.caption(
    f"Based on {n_sims_used:,} simulated brackets — "
    f"{teams_resolved} teams resolved from bracket data."
)

# ---------------------------------------------------------------------------
# Section 1 — Champion & Final Four
# ---------------------------------------------------------------------------

st.subheader("Projected Outcomes")
cols = st.columns(5)

champ_probs = adv_probs.get(projected_champion, [0.0] * 7)
champ_win_pct = f"{champ_probs[6] * 100:.1f}%"
cols[0].metric("Projected Champion", projected_champion, champ_win_pct)

for idx, f4_team in enumerate(projected_f4[:4]):
    f4_probs = adv_probs.get(f4_team, [0.0] * 7)
    champ_pct = f"{f4_probs[6] * 100:.1f}% to win title"
    cols[idx + 1].metric(f"Final Four #{idx + 1}", f4_team, champ_pct)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Upset Alerts
# ---------------------------------------------------------------------------

if upset_alerts:
    st.subheader("Upset Alerts")
    st.caption(
        "Games where the model gives the underdog 35% or greater chance — potential bracket busters"
    )

    alert_rows = []
    for a in upset_alerts:
        alert_rows.append({
            "Favorite": f"({a['fav_seed']}) {a['favorite']}",
            "Underdog": f"({a['dog_seed']}) {a['underdog']}",
            "Upset Prob": f"{a['upset_prob'] * 100:.1f}%",
            "AdjEM Gap": a.get("adj_em_gap", "N/A"),
            "Region": a.get("region", ""),
        })

    st.dataframe(
        pd.DataFrame(alert_rows),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Bracket by Region
# ---------------------------------------------------------------------------

projected_bracket: dict = result.get("projected_bracket", {})

# Collect the unique regions from the advancement_probs teams.
# We rebuild region membership from the bracket slot keys.
region_set: list = []
seen_regions: set = set()
for slot_key in projected_bracket:
    parts = slot_key.split("_")
    if len(parts) >= 2 and parts[0] not in seen_regions and parts[0] not in ("F4", "Champion"):
        seen_regions.add(parts[0])
        region_set.append(parts[0])

if region_set:
    st.subheader("Bracket by Region")

    round_labels = {
        "R64": "Round of 64",
        "R32": "Round of 32",
        "S16": "Sweet 16",
        "E8": "Elite 8",
    }

    for region_name in region_set:
        with st.expander(f"{region_name} Region", expanded=False):
            rows = []
            for rnd_key, rnd_label in round_labels.items():
                # Collect all slots for this region + round
                region_slots = sorted(
                    [k for k in projected_bracket if k.startswith(f"{region_name}_{rnd_key}")],
                    key=lambda k: k,
                )
                if not region_slots:
                    # E8 is stored as "{region}_E8" without a game index
                    single_key = f"{region_name}_{rnd_key}"
                    if single_key in projected_bracket:
                        region_slots = [single_key]

                for slot_key in region_slots:
                    team_name = projected_bracket[slot_key]
                    probs = adv_probs.get(team_name, [0.0] * 7)
                    # Choose the most meaningful probability for the round
                    prob_display_idx = {"R64": 0, "R32": 1, "S16": 2, "E8": 3}.get(rnd_key, 0)
                    win_prob = probs[prob_display_idx] * 100

                    label = f"**{team_name}**" if rnd_key == "E8" else team_name
                    rows.append({
                        "Round": rnd_label,
                        "Team": label,
                        "Advancement Prob": f"{win_prob:.1f}%",
                    })

            if rows:
                st.markdown(
                    pd.DataFrame(rows).to_markdown(index=False),
                    unsafe_allow_html=False,
                )
            else:
                st.info("No bracket data available for this region.")

    st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Full Advancement Probability Table
# ---------------------------------------------------------------------------

st.subheader("Full Advancement Probabilities")

if adv_probs:
    table_rows = []
    for team_name, probs in adv_probs.items():
        if len(probs) < 7:
            probs = probs + [0.0] * (7 - len(probs))

        table_rows.append({
            "Team": team_name,
            "R64": f"{probs[0] * 100:.1f}%",
            "R32": f"{probs[1] * 100:.1f}%",
            "S16": f"{probs[2] * 100:.1f}%",
            "E8": f"{probs[3] * 100:.1f}%",
            "F4": f"{probs[4] * 100:.1f}%",
            "Champ": f"{probs[5] * 100:.1f}%",
            "Win%_raw": probs[6],
            "Win%": f"{probs[6] * 100:.1f}%",
        })

    df = pd.DataFrame(table_rows).sort_values("Win%_raw", ascending=False)

    # Highlight rows where Win% > 5 %
    def highlight_contenders(row):
        if row["Win%_raw"] > 0.05:
            return ["background-color: #1a3a1a"] * len(row)
        return [""] * len(row)

    display_cols = ["Team", "R64", "R32", "S16", "E8", "F4", "Champ", "Win%"]
    styled = (
        df[display_cols + ["Win%_raw"]]
        .style.apply(highlight_contenders, axis=1)
    )

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No advancement probability data available.")
