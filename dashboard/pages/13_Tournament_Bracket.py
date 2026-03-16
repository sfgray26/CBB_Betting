"""
March Madness Bracket Simulator — interactive Streamlit UI.

Workflow:
1. Enter team names + seeds in the bracket tabs (or pull from DB)
2. Hit "Run Simulations"
3. Explore championship probabilities, Cinderella rankings, upset heat map
4. Optionally enter futures odds to find value bets
5. Download CSVs for any table

Calls backend/tournament/ directly (same Python process — no API hop needed).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import time
import math
import logging
from pathlib import Path
from io import StringIO

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Tournament Bracket | CBB Edge",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports (deferred so Streamlit still loads even if DB is unreachable)
# ---------------------------------------------------------------------------
try:
    from backend.tournament.matchup_predictor import TournamentTeam
    from backend.tournament.bracket_simulator import (
        run_monte_carlo, SimulationResults, R64_SEED_ORDER
    )
    from backend.tournament.cinderella_tracker import (
        cinderella_rankings, upset_heat_map,
    )
    from backend.tournament.futures_analyzer import (
        analyze_futures, american_to_implied, calculate_ev, implied_to_american,
    )
    TOURNAMENT_MODULE_OK = True
except ImportError as e:
    TOURNAMENT_MODULE_OK = False
    IMPORT_ERROR = str(e)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIONS = ["south", "east", "west", "midwest"]
REGION_LABELS = {"south": "South", "east": "East", "west": "West", "midwest": "Midwest"}
FF_PAIRINGS_DISPLAY = "South vs East  |  West vs Midwest"

# Default composite ratings by seed (approximate AdjEM — good starting point)
DEFAULT_RATINGS = {
    1: 26.0, 2: 21.5, 3: 18.0, 4: 15.5, 5: 13.0, 6: 11.0,
    7: 9.0, 8: 7.0, 9: 5.5, 10: 4.0, 11: 2.5, 12: 1.0,
    13: -1.0, 14: -2.5, 15: -4.5, 16: -7.0,
}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_bracket_state():
    """Initialize bracket state if not already set."""
    if "bracket_teams" not in st.session_state:
        st.session_state.bracket_teams = {
            region: [
                {
                    "seed": seed,
                    "name": "",
                    "composite_rating": DEFAULT_RATINGS.get(seed, 0.0),
                    "kp_adj_em": None,
                    "bt_adj_em": None,
                    "pace": 68.0,
                    "three_pt_rate": 0.35,
                    "def_efg_pct": 0.50,
                    "conference": "",
                    "tournament_exp": 0.70,
                }
                for seed in range(1, 17)
            ]
            for region in REGIONS
        }
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "bracket_for_results" not in st.session_state:
        st.session_state.bracket_for_results = None


def _get_bracket() -> dict:
    """Convert session state to {region: [TournamentTeam]} dict."""
    bracket = {}
    for region in REGIONS:
        teams = []
        for entry in st.session_state.bracket_teams[region]:
            name = entry["name"].strip()
            if not name:
                name = f"{REGION_LABELS[region]}-Seed{entry['seed']}"
            teams.append(TournamentTeam(
                name=name,
                seed=entry["seed"],
                region=region,
                composite_rating=entry["composite_rating"],
                kp_adj_em=entry.get("kp_adj_em"),
                bt_adj_em=entry.get("bt_adj_em"),
                pace=entry.get("pace", 68.0),
                three_pt_rate=entry.get("three_pt_rate", 0.35),
                def_efg_pct=entry.get("def_efg_pct", 0.50),
                conference=entry.get("conference", ""),
                tournament_exp=entry.get("tournament_exp", 0.70),
            ))
        bracket[region] = teams
    return bracket


def _pull_from_db():
    """Attempt to populate ratings from DB TeamProfile table."""
    try:
        from sqlalchemy.orm import Session
        from backend.models import engine, TeamProfile

        WEIGHT_KP = 0.51
        WEIGHT_BT = 0.49
        SEASON_YEAR = 2026
        found = 0
        missing = []

        with Session(engine) as session:
            for region in REGIONS:
                for entry in st.session_state.bracket_teams[region]:
                    name = entry["name"].strip()
                    if not name:
                        continue

                    kp = (session.query(TeamProfile)
                          .filter(TeamProfile.team_name == name,
                                  TeamProfile.season_year == SEASON_YEAR,
                                  TeamProfile.source == "kenpom")
                          .first())
                    bt = (session.query(TeamProfile)
                          .filter(TeamProfile.team_name == name,
                                  TeamProfile.season_year == SEASON_YEAR,
                                  TeamProfile.source == "barttorvik")
                          .first())

                    kp_em = kp.adj_em if kp else None
                    bt_em = bt.adj_em if bt else None

                    if kp_em is None and bt_em is None:
                        missing.append(name)
                        continue

                    ratings_sum = 0.0
                    weight_sum = 0.0
                    if kp_em is not None:
                        ratings_sum += kp_em * WEIGHT_KP
                        weight_sum += WEIGHT_KP
                    if bt_em is not None:
                        ratings_sum += bt_em * WEIGHT_BT
                        weight_sum += WEIGHT_BT

                    composite = (ratings_sum / weight_sum) * (weight_sum / 1.0)

                    entry["kp_adj_em"] = round(kp_em, 2) if kp_em else None
                    entry["bt_adj_em"] = round(bt_em, 2) if bt_em else None
                    entry["composite_rating"] = round(composite, 2)
                    if bt:
                        if bt.pace:
                            entry["pace"] = round(bt.pace, 1)
                        if bt.three_par:
                            entry["three_pt_rate"] = round(bt.three_par, 3)
                        if bt.def_efg_pct:
                            entry["def_efg_pct"] = round(bt.def_efg_pct, 3)
                    found += 1

        return found, missing
    except Exception as exc:
        return 0, [str(exc)]


def _load_2026_bracket_from_disk():
    """Load the pre-built 2026 bracket JSON from data/bracket_2026.json."""
    bracket_path = Path(__file__).resolve().parent.parent.parent / "data" / "bracket_2026.json"
    if not bracket_path.exists():
        raise FileNotFoundError(f"bracket_2026.json not found at {bracket_path}")
    with open(bracket_path, encoding="utf-8") as f:
        _load_bracket_json(f.read())


def _load_bracket_json(json_str: str):
    """Load bracket from uploaded JSON string into session state."""
    data = json.loads(json_str)
    for region in REGIONS:
        if region not in data:
            continue
        for entry in st.session_state.bracket_teams[region]:
            seed = entry["seed"]
            match = next((t for t in data[region] if t.get("seed") == seed), None)
            if match:
                entry["name"] = match.get("name", "")
                entry["composite_rating"] = match.get("composite_rating", DEFAULT_RATINGS.get(seed, 0.0))
                entry["kp_adj_em"] = match.get("kp_adj_em")
                entry["bt_adj_em"] = match.get("bt_adj_em")
                entry["pace"] = match.get("pace", 68.0)
                entry["three_pt_rate"] = match.get("three_pt_rate", 0.35)
                entry["def_efg_pct"] = match.get("def_efg_pct", 0.50)
                entry["conference"] = match.get("conference", "")
                entry["tournament_exp"] = match.get("tournament_exp", 0.70)


def _export_bracket_json() -> str:
    """Serialize current bracket state to JSON string."""
    out = {}
    for region in REGIONS:
        out[region] = [dict(e) for e in st.session_state.bracket_teams[region]]
    return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
_init_bracket_state()

st.title("March Madness Bracket Simulator")
st.caption(f"Monte Carlo simulation engine | V9.1 ratings | Final Four: {FF_PAIRINGS_DISPLAY}")

if not TOURNAMENT_MODULE_OK:
    st.error(f"Tournament module import failed: {IMPORT_ERROR}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — simulation controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Simulation Settings")
    n_sims = st.select_slider(
        "Simulations",
        options=[1000, 5000, 10000, 25000, 50000],
        value=10000,
        help="More sims = higher accuracy but slower. 10k runs in ~30-60 sec.",
    )
    n_workers = st.slider("Parallel workers", 1, 8, 4)
    rand_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.header("Import / Export")

    if st.button("Load 2026 Bracket", use_container_width=True, type="primary",
                 help="Load the official 2026 NCAA Tournament bracket with pre-built composite ratings"):
        try:
            _load_2026_bracket_from_disk()
            st.success("2026 bracket loaded with ratings.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load bracket: {exc}")

    st.caption("Or upload a custom bracket JSON:")
    uploaded = st.file_uploader("Load bracket JSON", type="json", key="bracket_upload")
    if uploaded:
        try:
            _load_bracket_json(uploaded.read().decode())
            st.success("Bracket loaded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON: {exc}")

    if st.button("Export bracket JSON", use_container_width=True):
        st.download_button(
            "Download bracket_2026.json",
            data=_export_bracket_json(),
            file_name="bracket_2026.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Section 1: Bracket Input
# ---------------------------------------------------------------------------
st.subheader("1. Enter the Bracket")
st.caption("Load the 2026 bracket from the sidebar, then optionally refresh ratings from the DB if it is reachable.")

col_load, col_pull, col_reset, col_spacer = st.columns([2, 2, 2, 4])
with col_load:
    if st.button("Load 2026 Bracket", use_container_width=True, type="primary"):
        try:
            _load_2026_bracket_from_disk()
            st.success("Loaded.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

with col_pull:
    if st.button("Refresh from DB", use_container_width=True, type="secondary",
                 help="Overwrite ratings from team_profiles table. Requires local DB connection."):
        with st.spinner("Looking up team ratings..."):
            found, missing = _pull_from_db()
        if found > 0:
            st.success(f"Updated {found} teams.")
            st.rerun()
        elif missing and len(missing) == 1 and ("password" in missing[0].lower() or "connect" in missing[0].lower() or "refused" in missing[0].lower()):
            st.warning("DB not reachable locally — bracket JSON ratings are already pre-built and ready to use.")
        else:
            st.warning(f"No ratings found. Teams missing: {', '.join(missing[:5])}")

with col_reset:
    if st.button("Reset bracket", use_container_width=True):
        del st.session_state["bracket_teams"]
        st.session_state.sim_results = None
        st.rerun()

# Region tabs
region_tabs = st.tabs([REGION_LABELS[r] for r in REGIONS])

for tab, region in zip(region_tabs, REGIONS):
    with tab:
        # Display as a data editor with key columns
        entries = st.session_state.bracket_teams[region]

        # Build editable DataFrame
        df_edit = pd.DataFrame([
            {
                "Seed": e["seed"],
                "Team Name": e["name"],
                "Composite Rating": e["composite_rating"],
                "KP AdjEM": e["kp_adj_em"] if e["kp_adj_em"] is not None else "",
                "BT AdjEM": e["bt_adj_em"] if e["bt_adj_em"] is not None else "",
                "Pace": e["pace"],
                "3PT Rate": e["three_pt_rate"],
                "Def eFG%": e["def_efg_pct"],
                "Conference": e["conference"],
                "Tourney Exp": e["tournament_exp"],
            }
            for e in entries
        ])

        edited = st.data_editor(
            df_edit,
            key=f"editor_{region}",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Seed": st.column_config.NumberColumn("Seed", disabled=True, width="small"),
                "Team Name": st.column_config.TextColumn("Team Name", width="medium"),
                "Composite Rating": st.column_config.NumberColumn(
                    "Rating", format="%.1f",
                    help="V9.1 composite AdjEM (51% KenPom + 49% BartTorvik). Range: ~-7 (16-seed) to ~26 (1-seed)"
                ),
                "KP AdjEM": st.column_config.NumberColumn("KP AdjEM", format="%.1f"),
                "BT AdjEM": st.column_config.NumberColumn("BT AdjEM", format="%.1f"),
                "Pace": st.column_config.NumberColumn("Pace", format="%.1f", width="small"),
                "3PT Rate": st.column_config.NumberColumn("3PT%", format="%.3f", width="small"),
                "Def eFG%": st.column_config.NumberColumn("Def eFG%", format="%.3f", width="small"),
                "Conference": st.column_config.TextColumn("Conf", width="small"),
                "Tourney Exp": st.column_config.NumberColumn(
                    "Exp", format="%.2f", width="small",
                    help="Fraction of minutes from returning tournament-experienced players (0-1)"
                ),
            },
        )

        # Write edits back to session state
        for i, row in edited.iterrows():
            e = entries[i]
            e["name"] = str(row["Team Name"]).strip()
            e["composite_rating"] = float(row["Composite Rating"]) if row["Composite Rating"] else DEFAULT_RATINGS.get(e["seed"], 0.0)
            e["kp_adj_em"] = float(row["KP AdjEM"]) if row["KP AdjEM"] not in ("", None) else None
            e["bt_adj_em"] = float(row["BT AdjEM"]) if row["BT AdjEM"] not in ("", None) else None
            e["pace"] = float(row["Pace"]) if row["Pace"] else 68.0
            e["three_pt_rate"] = float(row["3PT Rate"]) if row["3PT Rate"] else 0.35
            e["def_efg_pct"] = float(row["Def eFG%"]) if row["Def eFG%"] else 0.50
            e["conference"] = str(row["Conference"]).strip() if row["Conference"] else ""
            e["tournament_exp"] = float(row["Tourney Exp"]) if row["Tourney Exp"] else 0.70

# ---------------------------------------------------------------------------
# Section 2: Run Simulations
# ---------------------------------------------------------------------------
st.divider()
st.subheader("2. Run Simulations")

run_col, status_col = st.columns([2, 5])
with run_col:
    run_clicked = st.button(
        f"Run {n_sims:,} Simulations",
        type="primary",
        use_container_width=True,
    )

if run_clicked:
    bracket = _get_bracket()
    n_named = sum(
        1 for teams in bracket.values()
        for t in teams
        if not t.name.startswith(f"{REGION_LABELS[t.region]}-Seed")
    )

    with status_col:
        if n_named == 0:
            st.warning("No team names entered — simulating with seed-based placeholders.")
        else:
            st.info(f"{n_named} teams named. Running {n_sims:,} simulations...")

    with st.spinner(f"Simulating {n_sims:,} tournaments across {n_workers} workers..."):
        t0 = time.time()
        results = run_monte_carlo(
            bracket,
            n_sims=n_sims,
            n_workers=n_workers,
            base_seed=int(rand_seed),
        )
        elapsed = time.time() - t0

    st.session_state.sim_results = results
    st.session_state.bracket_for_results = bracket
    st.success(f"Done in {elapsed:.1f}s ({n_sims / elapsed:.0f} sims/sec). Scroll down for results.")
    st.rerun()

# ---------------------------------------------------------------------------
# Section 3: Results (only shown after a run)
# ---------------------------------------------------------------------------
results: SimulationResults = st.session_state.sim_results
bracket_snapshot = st.session_state.bracket_for_results

if results is None:
    st.info("Enter the bracket above and click **Run Simulations** to see projections.")
    st.stop()

st.divider()
st.subheader("3. Results")
st.caption(f"Based on {results.n_sims:,} simulated tournaments | Avg championship margin: {results.avg_championship_margin:.1f} pts | Avg upsets/tournament: {results.avg_upsets_per_tournament:.1f}")

# --- 3a: Championship Probabilities ---
all_teams_info = [
    {"name": t.name, "seed": t.seed, "region": t.region}
    for teams in bracket_snapshot.values()
    for t in teams
]
seed_map = {t["name"]: t["seed"] for t in all_teams_info}
region_map = {t["name"]: t["region"] for t in all_teams_info}

champ_tab, ff_tab, cinderella_tab, upset_tab, futures_tab = st.tabs([
    "Championship", "Final Four", "Cinderellas", "Upset Heat Map", "Futures Value"
])

with champ_tab:
    st.markdown("**Championship probabilities — all 68 teams**")

    sorted_champ = sorted(results.championship.items(), key=lambda x: -x[1])
    df_champ = pd.DataFrame([
        {
            "Team": t,
            "Seed": seed_map.get(t, "?"),
            "Region": REGION_LABELS.get(region_map.get(t, ""), ""),
            "Champion %": round(p * 100, 1),
            "Final Four %": round(results.final_four.get(t, 0) * 100, 1),
            "Elite 8 %": round(results.elite_eight.get(t, 0) * 100, 1),
            "Sweet 16 %": round(results.sweet_sixteen.get(t, 0) * 100, 1),
        }
        for t, p in sorted_champ
    ])

    # Bar chart of top 20
    top20 = df_champ.head(20)
    st.bar_chart(
        top20.set_index("Team")["Champion %"],
        use_container_width=True,
        color="#1f77b4",
    )

    st.dataframe(
        df_champ,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Champion %": st.column_config.ProgressColumn(
                "Champion %", format="%.1f%%", min_value=0, max_value=df_champ["Champion %"].max()
            ),
        },
    )

    csv = df_champ.to_csv(index=False)
    st.download_button("Download championship_probs.csv", csv, "championship_probs.csv", "text/csv")

with ff_tab:
    st.markdown("**Milestone probabilities — all teams**")

    all_names = set(results.championship) | set(results.final_four) | set(results.sweet_sixteen)
    df_ff = pd.DataFrame([
        {
            "Team": t,
            "Seed": seed_map.get(t, "?"),
            "Region": REGION_LABELS.get(region_map.get(t, ""), ""),
            "R32 %": round(results.round_of_32.get(t, 0) * 100, 1),
            "S16 %": round(results.sweet_sixteen.get(t, 0) * 100, 1),
            "E8 %": round(results.elite_eight.get(t, 0) * 100, 1),
            "F4 %": round(results.final_four.get(t, 0) * 100, 1),
            "Runner-up %": round(results.runner_up.get(t, 0) * 100, 1),
            "Champion %": round(results.championship.get(t, 0) * 100, 1),
        }
        for t in sorted(all_names, key=lambda x: -results.championship.get(x, 0))
    ]).reset_index(drop=True)

    st.dataframe(df_ff, use_container_width=True, hide_index=True)

    csv = df_ff.to_csv(index=False)
    st.download_button("Download milestone_probs.csv", csv, "milestone_probs.csv", "text/csv")

with cinderella_tab:
    st.markdown("**Double-digit seeds with real deep-run probability**")
    st.caption("Cinderella Score = seed x P(S16) x 10 — higher is a bigger story")

    cinderellas = cinderella_rankings(results, all_teams_info, min_seed=10, min_s16_prob=0.005)

    if not cinderellas:
        st.info("No double-digit seeds with >0.5% Sweet 16 probability. Try more sims or adjust ratings.")
    else:
        df_cind = pd.DataFrame([
            {
                "Team": c.team,
                "Seed": c.seed,
                "Region": REGION_LABELS.get(c.region, ""),
                "P(R32) %": round(c.p_round_of_32 * 100, 1),
                "P(S16) %": round(c.p_sweet_sixteen * 100, 1),
                "P(E8) %": round(c.p_elite_eight * 100, 1),
                "P(F4) %": round(c.p_final_four * 100, 1),
                "Cinderella Score": round(c.cinderella_score, 2),
            }
            for c in cinderellas
        ])

        # Highlight top candidate
        top = cinderellas[0]
        st.metric(
            label=f"Top Cinderella: #{top.seed} {top.team}",
            value=f"{top.p_sweet_sixteen * 100:.1f}% to reach Sweet 16",
            delta=f"{top.p_elite_eight * 100:.1f}% Elite 8 | {top.p_final_four * 100:.1f}% Final Four",
        )

        st.dataframe(
            df_cind,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P(S16) %": st.column_config.ProgressColumn(
                    "P(S16) %", format="%.1f%%", min_value=0, max_value=df_cind["P(S16) %"].max()
                ),
                "Cinderella Score": st.column_config.NumberColumn("Score", format="%.2f"),
            },
        )

        csv = df_cind.to_csv(index=False)
        st.download_button("Download cinderella_rankings.csv", csv, "cinderella_rankings.csv", "text/csv")

with upset_tab:
    st.markdown("**R64 upset probability for every first-round matchup**")
    st.caption("Based on V9.1 ratings + historical seed upset rates (2000-2024)")

    upsets = upset_heat_map(bracket_snapshot)

    if not upsets:
        st.info("No matchup data — run simulations first.")
    else:
        RISK_COLOR = {"HIGH": "red", "MED": "orange", "LOW": "green"}

        df_upsets = pd.DataFrame([
            {
                "Region": REGION_LABELS.get(m.region, m.region),
                "Matchup": f"#{m.favorite_seed} {m.favorite} vs #{m.underdog_seed} {m.underdog}",
                "Upset %": round(m.upset_probability * 100, 1),
                "Est. Margin": m.margin_estimate,
                "Risk": m.risk_level,
            }
            for m in upsets
        ])

        n_high = sum(1 for m in upsets if m.risk_level == "HIGH")
        n_med = sum(1 for m in upsets if m.risk_level == "MED")
        m1, m2, m3 = st.columns(3)
        m1.metric("HIGH risk matchups", n_high, help=">35% upset probability")
        m2.metric("MED risk matchups", n_med, help="20-35% upset probability")
        m3.metric("LOW risk matchups", len(upsets) - n_high - n_med)

        st.dataframe(
            df_upsets,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Upset %": st.column_config.ProgressColumn(
                    "Upset %", format="%.1f%%", min_value=0, max_value=55
                ),
                "Est. Margin": st.column_config.NumberColumn("Margin", format="%+.1f"),
            },
        )

        csv = df_upsets.to_csv(index=False)
        st.download_button("Download upset_heatmap_r64.csv", csv, "upset_heatmap_r64.csv", "text/csv")

with futures_tab:
    st.markdown("**Enter market futures odds to find value bets**")
    st.caption("American odds format: +1200 = 12/1 underdog, -150 = favorite. Leave blank to skip.")

    # Dynamic futures table — user inputs odds
    all_named_teams = sorted(
        [t for t in results.championship],
        key=lambda x: -results.championship.get(x, 0)
    )[:30]  # Top 30 most likely champs

    st.markdown("Enter American odds for Championship and/or Final Four futures:")

    if "futures_input" not in st.session_state:
        st.session_state.futures_input = {t: {"championship": "", "final_four": ""} for t in all_named_teams}

    futures_df = pd.DataFrame([
        {
            "Team": t,
            "Model Champ %": f"{results.championship.get(t, 0) * 100:.1f}%",
            "Model F4 %": f"{results.final_four.get(t, 0) * 100:.1f}%",
            "Champ Odds (e.g. +1200)": st.session_state.futures_input.get(t, {}).get("championship", ""),
            "F4 Odds (e.g. +450)": st.session_state.futures_input.get(t, {}).get("final_four", ""),
        }
        for t in all_named_teams
    ])

    edited_futures = st.data_editor(
        futures_df,
        key="futures_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Team": st.column_config.TextColumn("Team", disabled=True),
            "Model Champ %": st.column_config.TextColumn("Model Champ %", disabled=True, width="small"),
            "Model F4 %": st.column_config.TextColumn("Model F4 %", disabled=True, width="small"),
            "Champ Odds (e.g. +1200)": st.column_config.TextColumn("Champ Odds", width="medium"),
            "F4 Odds (e.g. +450)": st.column_config.TextColumn("F4 Odds", width="medium"),
        },
    )

    if st.button("Analyze Futures Value", type="primary"):
        # Parse the edited table into market_odds dict
        market_odds = {}
        parse_errors = []

        for _, row in edited_futures.iterrows():
            team = row["Team"]
            team_odds = {}

            for col, mkt in [
                ("Champ Odds (e.g. +1200)", "championship"),
                ("F4 Odds (e.g. +450)", "final_four"),
            ]:
                val = str(row.get(col, "")).strip().replace(" ", "")
                if not val:
                    continue
                try:
                    team_odds[mkt] = int(val.replace("+", ""))
                except ValueError:
                    parse_errors.append(f"{team} {mkt}: '{val}' is not a valid integer")

            if team_odds:
                market_odds[team] = team_odds

        if parse_errors:
            for err in parse_errors:
                st.warning(f"Parse error: {err}")

        if not market_odds:
            st.info("Enter at least one odds value above to find value bets.")
        else:
            value_bets = analyze_futures(results, market_odds, min_ev_pct=0.02)

            if not value_bets:
                st.info("No value bets found (EV < 2% on all entries). Market may be fairly priced.")
            else:
                n_bet = sum(1 for b in value_bets if b.recommendation == "BET")
                n_consider = sum(1 for b in value_bets if b.recommendation == "CONSIDER")

                c1, c2, c3 = st.columns(3)
                c1.metric("BET recommendations", n_bet)
                c2.metric("CONSIDER recommendations", n_consider)
                c3.metric("Best EV", f"{value_bets[0].ev_pct:.1f}%", value_bets[0].team)

                df_val = pd.DataFrame([
                    {
                        "Team": b.team,
                        "Market": b.market.replace("_", " ").title(),
                        "Odds": f"+{b.american_odds}" if b.american_odds > 0 else str(b.american_odds),
                        "Fair Odds": f"+{b.fair_american_odds}" if b.fair_american_odds > 0 else str(b.fair_american_odds),
                        "Model %": f"{b.model_prob * 100:.1f}%",
                        "Market %": f"{b.market_implied_prob * 100:.1f}%",
                        "Edge": f"+{b.edge_pct:.1f}pp",
                        "EV": f"+{b.ev_pct:.1f}%",
                        "Rec": b.recommendation,
                    }
                    for b in value_bets
                ])

                st.dataframe(
                    df_val,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "EV": st.column_config.TextColumn("EV", width="small"),
                        "Rec": st.column_config.TextColumn("Rec", width="small"),
                    },
                )

                csv = df_val.to_csv(index=False)
                st.download_button(
                    "Download futures_value.csv", csv, "futures_value.csv", "text/csv"
                )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Simulation uses V9.1 composite ratings (51% KenPom / 49% BartTorvik) with round-specific variance "
    "multipliers (R64: 1.12x SD) and historical seed upset rate blending (80% model / 20% historical). "
    "Final Four: South vs East, West vs Midwest."
)
