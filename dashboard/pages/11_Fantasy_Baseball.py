"""
Fantasy Baseball Dashboard — Treemendous League (Yahoo ID 72586)
12-team H2H One Win · 18 categories · Snake Draft · Keeper League

Pages:
  - Keeper Evaluator  (active: deadline Mar 20)
  - Draft Board       (active: Mar 22-23)
  - My Roster         (season view)
  - Waiver Wire       (in-season)
  - Trade Analyzer    (in-season)
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure backend imports work when running from dashboard/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from backend.fantasy_baseball.keeper_engine import (
    KeeperEngine,
    PlayerProjection,
    CategoryValueEngine,
    soto_2026_projection,
)
from backend.fantasy_baseball.player_board import get_board

st.set_page_config(
    page_title="Fantasy Baseball — Treemendous",
    page_icon="⚾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚾ Fantasy Baseball — Treemendous League")
st.caption("Yahoo ID 72586 · 12 teams · H2H One Win · 18 categories · Keeper League")

col1, col2, col3 = st.columns(3)
col1.metric("Keeper Deadline", "Mar 20 @ 3am EDT", delta=None)
col2.metric("Draft Day", "Mar 23 @ 7:30am EDT", delta=None)
col3.metric("Draft Format", "Snake · 90-sec clock", delta=None)

st.divider()

# ---------------------------------------------------------------------------
# Navigation tabs
# ---------------------------------------------------------------------------
tab_keeper, tab_draft, tab_roster, tab_waiver, tab_trade, tab_setup = st.tabs([
    "🔑 Keeper Evaluator",
    "📋 Draft Board",
    "👥 My Roster",
    "📈 Waiver Wire",
    "↔️ Trade Analyzer",
    "⚙️ Setup & Auth",
])

# ===========================================================================
# TAB: Keeper Evaluator
# ===========================================================================
with tab_keeper:
    st.header("Keeper Evaluator")
    st.info(
        "**Deadline: Fri Mar 20 @ 3:00am EDT**  \n"
        "Keeper surplus = projected z-score minus the expected value of a player "
        "drafted at the same round. Positive surplus = worth keeping."
    )

    st.subheader("League Category Reminder")
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        st.markdown("""
**Batting (9 cats)**
| Cat | Direction | Note |
|-----|-----------|------|
| R   | ✅ Higher better | |
| H   | ✅ Higher better | Standalone — rewards contact |
| HR  | ✅ Higher better | |
| RBI | ✅ Higher better | |
| K   | ❌ Lower better | Strikeouts HURT batters |
| TB  | ✅ Higher better | SLG-correlated |
| AVG | ✅ Higher better | |
| OPS | ✅ Higher better | Most valuable rate stat |
| NSB | ✅ Higher better | SB−CS · Scarcest cat |
""")
    with cat_col2:
        st.markdown("""
**Pitching (9 cats)**
| Cat  | Direction | Note |
|------|-----------|------|
| W    | ✅ Higher better | |
| L    | ❌ Lower better | Losses HURT |
| HR   | ❌ Lower better | HR allowed |
| K    | ✅ Higher better | |
| ERA  | ❌ Lower better | |
| WHIP | ❌ Lower better | |
| K/9  | ✅ Higher better | Double K value |
| QS   | ✅ Higher better | |
| NSV  | ✅ Higher better | SV−BS · Very scarce |
""")

    st.divider()

    # --- Keeper entry form ---
    st.subheader("Add Your Keeper Candidates")
    st.caption("Enter each rostered player you're considering keeping. The engine will rank them by surplus value.")

    with st.expander("Pre-loaded: Juan Soto (adjust keeper round cost)", expanded=True):
        soto = soto_2026_projection()
        soto.keeper_round_cost = st.slider(
            "What round does keeping Soto cost you?",
            min_value=1, max_value=5, value=1,
            help="In Yahoo keeper leagues, keeping a player typically costs their draft round from last year + 1"
        )

        st.markdown(f"""
**Juan Soto — 2026 Projection (Steamer/ZiPS consensus)**

| Stat | Projection | Note |
|------|-----------|------|
| PA   | {soto.pa} | Full season |
| HR   | {soto.hr} | Elite power |
| R/RBI | {soto.r} / {soto.rbi} | Top-tier run producer |
| K%   | {soto.k_bat/soto.pa:.1%} | Low K rate for a power bat |
| AVG  | {soto.avg:.3f} | |
| OPS  | {soto.ops:.3f} | Elite |
| NSB  | {soto.nsb} | Modest base stealing |
""")

    st.markdown("---")
    st.subheader("Add More Keepers Manually")
    st.info("Use the form below to add other players on your roster you're considering keeping.")

    if "extra_keepers" not in st.session_state:
        st.session_state.extra_keepers = []

    with st.form("add_keeper_form"):
        k_col1, k_col2, k_col3 = st.columns(3)
        with k_col1:
            k_name = st.text_input("Player Name", placeholder="e.g. Yordan Alvarez")
            k_team = st.text_input("MLB Team", placeholder="e.g. HOU")
            k_type = st.selectbox("Player Type", ["batter", "pitcher"])
            k_round = st.number_input("Keeper Round Cost", min_value=1, max_value=23, value=5)
        with k_col2:
            if k_type == "batter":
                k_pa = st.number_input("PA", 0, 750, 550)
                k_r = st.number_input("R", 0, 150, 70)
                k_h = st.number_input("H", 0, 250, 140)
                k_hr = st.number_input("HR", 0, 60, 20)
                k_rbi = st.number_input("RBI", 0, 150, 70)
            else:
                k_ip = st.number_input("IP", 0.0, 250.0, 170.0)
                k_w = st.number_input("W", 0, 25, 10)
                k_l = st.number_input("L", 0, 15, 7)
                k_qs = st.number_input("QS", 0, 35, 18)
                k_sv = st.number_input("SV", 0, 50, 0)
        with k_col3:
            if k_type == "batter":
                k_k = st.number_input("K (batter Ks)", 0, 250, 100)
                k_tb = st.number_input("TB", 0, 400, 220)
                k_avg = st.number_input("AVG", 0.150, 0.400, 0.265, step=0.001, format="%.3f")
                k_ops = st.number_input("OPS", 0.500, 1.200, 0.780, step=0.001, format="%.3f")
                k_nsb = st.number_input("NSB (SB-CS)", -10, 80, 5)
            else:
                k_k_p = st.number_input("K (pitcher)", 0, 350, 160)
                k_era = st.number_input("ERA", 0.0, 8.0, 4.00, step=0.01, format="%.2f")
                k_whip = st.number_input("WHIP", 0.5, 2.5, 1.25, step=0.01, format="%.2f")
                k_k9 = st.number_input("K/9", 0.0, 15.0, 8.5, step=0.1)
                k_nsv = st.number_input("NSV (SV-BS)", -10, 50, 0)
                k_hr_p = st.number_input("HR allowed", 0, 50, 18)

        submitted = st.form_submit_button("Add Keeper Candidate")
        if submitted and k_name:
            p = PlayerProjection(
                name=k_name, yahoo_player_key="manual",
                team=k_team, positions=[], player_type=k_type,
                keeper_round_cost=k_round,
            )
            if k_type == "batter":
                p.pa = k_pa; p.r = k_r; p.h = k_h; p.hr = k_hr; p.rbi = k_rbi
                p.k_bat = k_k; p.tb = k_tb; p.avg = k_avg; p.ops = k_ops; p.nsb = k_nsb
                p.slg = k_ops - 0.320  # rough approximation if OBP unknown
            else:
                p.ip = k_ip; p.w = k_w; p.l = k_l; p.qs = k_qs
                p.sv = k_sv; p.k_pit = k_k_p; p.era = k_era; p.whip = k_whip
                p.k9 = k_k9; p.nsv = k_nsv; p.hr_pit = k_hr_p
                p.bs = max(0, k_sv - k_nsv)  # infer BS from SV and NSV
            st.session_state.extra_keepers.append(p)
            st.success(f"Added {k_name}")

    # Show added keepers
    if st.session_state.extra_keepers:
        st.write(f"**{len(st.session_state.extra_keepers)} extra keepers queued**")
        for ep in st.session_state.extra_keepers:
            st.write(f"  • {ep.name} ({ep.team}) — {ep.player_type} — Round {ep.keeper_round_cost} cost")
        if st.button("Clear all extra keepers"):
            st.session_state.extra_keepers = []

    st.divider()

    # --- Run evaluation ---
    if st.button("🔍 Run Keeper Evaluation", type="primary"):
        all_keepers = [soto] + st.session_state.extra_keepers

        # Build a representative player pool for z-score context
        # These are approximate 2026 average starters for z-score baseline
        pool = _build_baseline_pool()
        pool.extend(all_keepers)

        engine = KeeperEngine(pool)
        report = engine.evaluate_roster(all_keepers)

        st.subheader("Keeper Rankings — Surplus Value Analysis")

        import pandas as pd
        rows = []
        for p in report.players:
            surplus_str = f"+{p.keeper_surplus:.2f}" if p.keeper_surplus >= 0 else f"{p.keeper_surplus:.2f}"
            verdict = "KEEP ✓" if p.keeper_surplus > 0.5 else ("BORDERLINE" if p.keeper_surplus > 0 else "RELEASE ✗")
            rows.append({
                "Player": p.name,
                "Type": p.player_type.title(),
                "Team": p.team,
                "Round Cost": p.keeper_round_cost,
                "Z-Score": f"{p.z_score:+.2f}",
                "Surplus": surplus_str,
                "Verdict": verdict,
                "Notes": " | ".join(p.notes[:2]),
            })
        df = pd.DataFrame(rows)

        def color_verdict(val):
            if "KEEP" in val:
                return "background-color: #1a472a; color: white"
            elif "BORDERLINE" in val:
                return "background-color: #7b6d00; color: white"
            else:
                return "background-color: #6b1a1a; color: white"

        st.dataframe(
            df.style.applymap(color_verdict, subset=["Verdict"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Category Contribution Breakdown")
        for p in report.players:
            if p.category_contributions:
                with st.expander(f"{p.name} — category breakdown"):
                    cat_data = [
                        {"Category": cat, "Weighted Z-Score": f"{val:+.3f}"}
                        for cat, val in sorted(
                            p.category_contributions.items(),
                            key=lambda x: abs(x[1]), reverse=True
                        )
                    ]
                    st.dataframe(pd.DataFrame(cat_data), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB: Draft Board — Interactive Draft Assistant
# ===========================================================================

# ---------------------------------------------------------------------------
# Roster slot definition for Treemendous (23 total)
# ---------------------------------------------------------------------------
ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
    "OF": 3, "Util": 1,
    "SP": 5, "RP": 2, "P": 1,
    "BN": 5,
}
TOTAL_PICKS = sum(ROSTER_SLOTS.values())  # 23

# Positions that can fill each slot
SLOT_ELIGIBLE = {
    "C": ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "OF": ["OF", "LF", "CF", "RF"],
    "Util": ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH"],
    "SP": ["SP"],
    "RP": ["RP"],
    "P": ["SP", "RP"],
    "BN": ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH", "SP", "RP"],
}


def _can_fill_slot(player_positions: list, slot: str) -> bool:
    eligible = SLOT_ELIGIBLE.get(slot, [])
    return any(pos in eligible for pos in player_positions)


def _compute_remaining_slots(drafted_players: list) -> dict:
    """Return remaining open slots given the drafted player list."""
    remaining = dict(ROSTER_SLOTS)
    for p in drafted_players:
        pos_list = p.get("positions", [])
        # Greedily fill the most specific slot first
        slot_order = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP", "P", "Util", "BN"]
        for slot in slot_order:
            if remaining.get(slot, 0) > 0 and _can_fill_slot(pos_list, slot):
                remaining[slot] -= 1
                break
    return remaining


def _position_needed(remaining_slots: dict) -> list:
    """Return position groups that still have open roster slots (excluding BN)."""
    needed = []
    for slot, count in remaining_slots.items():
        if slot == "BN":
            continue
        if count > 0:
            needed.append(slot)
    return needed


def _player_fills_need(player: dict, needed_slots: list) -> bool:
    pos_list = player.get("positions", [])
    for slot in needed_slots:
        if _can_fill_slot(pos_list, slot):
            return True
    return False


def _get_proj_stat(player: dict, stat: str, default=0):
    return player.get("proj", {}).get(stat, default)


def _load_draft_board() -> list:
    """Load board, preferring CSV projections over hardcoded fallback."""
    try:
        from backend.fantasy_baseball.projections_loader import load_full_board
        csv_board = load_full_board()
        if csv_board and len(csv_board) >= 100:
            return csv_board
    except Exception:
        pass
    try:
        return get_board()
    except Exception:
        return []


with tab_draft:
    st.header("Draft Board Assistant")

    st.info(
        "**Draft: Mon Mar 23 @ 7:30am EDT · 90-sec clock · 12-team Snake · 23 rounds**  \n"
        "Enter your pick number below to get recommendations. "
        "Use **Draft Player** to cross off players as the draft progresses."
    )

    # --- Session state init ---
    if "drafted" not in st.session_state:
        st.session_state["drafted"] = []  # list of player dicts
    if "draft_pick_num" not in st.session_state:
        st.session_state["draft_pick_num"] = 1

    # --- Load board (cached via session state to avoid re-loading on every rerender) ---
    if "draft_board_cache" not in st.session_state:
        with st.spinner("Loading 2026 projection board..."):
            st.session_state["draft_board_cache"] = _load_draft_board()

    full_board = st.session_state["draft_board_cache"]

    if not full_board:
        st.error("Could not load player board. Verify data/projections/ CSV files exist.")
        st.stop()

    drafted_ids = {p["id"] for p in st.session_state["drafted"]}
    available = [p for p in full_board if p["id"] not in drafted_ids]

    st.caption(
        f"{len(full_board)} players loaded  |  "
        f"{len(st.session_state['drafted'])} drafted  |  "
        f"{len(available)} remaining"
    )

    # -----------------------------------------------------------------------
    # Section 1: Pick Input + Recommendations
    # -----------------------------------------------------------------------
    st.subheader("My Pick")
    rec_col1, rec_col2 = st.columns([1, 3])
    with rec_col1:
        my_pick = st.number_input(
            "My pick number (overall)",
            min_value=1, max_value=276, value=st.session_state["draft_pick_num"],
            step=1, key="pick_num_input",
            help="Enter your next overall pick number (e.g. pick 1, 25, 49, ...)"
        )
        st.session_state["draft_pick_num"] = int(my_pick)

    remaining_slots = _compute_remaining_slots(st.session_state["drafted"])
    needed_slots = _position_needed(remaining_slots)
    picks_made = len(st.session_state["drafted"])
    picks_remaining = TOTAL_PICKS - picks_made

    with rec_col2:
        slot_cols = st.columns(len(remaining_slots))
        for col, (slot, cnt) in zip(slot_cols, remaining_slots.items()):
            color = "normal" if cnt > 0 else "off"
            col.metric(slot, cnt, delta=None)

    st.markdown("---")

    # Top 3 recommendations
    st.subheader("Recommended Picks")
    st.caption(
        f"Round ~{(my_pick - 1) // 12 + 1} · "
        f"Best available by z-score, weighted toward position need"
    )

    # Sort available by z-score descending
    available_sorted = sorted(available, key=lambda p: p.get("z_score", 0), reverse=True)

    # Build recommendations: prioritize positional need, then pure best available
    need_recs = [p for p in available_sorted if _player_fills_need(p, needed_slots)][:3]
    bva_recs = [p for p in available_sorted if p not in need_recs][:3]

    def _rec_card(player: dict, label: str):
        proj = player.get("proj", {})
        pos_str = "/".join(player.get("positions", [])[:3])
        z = player.get("z_score", 0)
        adp = player.get("adp", 999)
        adp_str = f"ADP {adp:.0f}" if adp < 999 else "Undrafted"

        if player["type"] == "batter":
            stat_line = (
                f"HR {proj.get('hr', 0):.0f} | "
                f"R {proj.get('r', 0):.0f} | "
                f"RBI {proj.get('rbi', 0):.0f} | "
                f"SB {proj.get('nsb', proj.get('sb', 0)):.0f} | "
                f"AVG {proj.get('avg', 0):.3f}"
            )
        else:
            stat_line = (
                f"W {proj.get('w', 0):.0f} | "
                f"K {proj.get('k_pit', 0):.0f} | "
                f"ERA {proj.get('era', 0):.2f} | "
                f"WHIP {proj.get('whip', 0):.2f} | "
                f"SV {proj.get('sv', proj.get('nsv', 0)):.0f}"
            )

        with st.container():
            st.markdown(
                f"**{player['name']}** ({player['team']}) — {pos_str}  \n"
                f"Z: {z:+.2f} | {adp_str} | Tier {player.get('tier', '?')}  \n"
                f"{stat_line}"
            )
            if st.button(f"Draft {player['name']}", key=f"draft_btn_{player['id']}_{label}"):
                if player["id"] not in drafted_ids:
                    st.session_state["drafted"].append(player)
                    st.session_state["draft_pick_num"] = int(my_pick) + 1
                    st.rerun()

    if need_recs:
        st.markdown("**Position Need Picks**")
        need_cols = st.columns(min(3, len(need_recs)))
        for col, p in zip(need_cols, need_recs):
            with col:
                _rec_card(p, "need")

    st.markdown("**Best Available (by Z-Score)**")
    bva_cols = st.columns(min(3, len(bva_recs)))
    for col, p in zip(bva_cols, bva_recs):
        with col:
            _rec_card(p, "bva")

    # -----------------------------------------------------------------------
    # Section 2: Full Available Board with Filters
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Available Player Board")

    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    with col_f1:
        pos_opts = ["All", "SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "DH"]
        pos_filter = st.selectbox("Position", pos_opts, key="db2_pos_filter")
    with col_f2:
        type_opts = ["All", "Batter", "Pitcher"]
        type_filter = st.selectbox("Type", type_opts, key="db2_type_filter")
    with col_f3:
        tier_opts_vals = sorted({p["tier"] for p in available if p.get("tier", 0) > 0})
        tier_opts = ["All"] + [str(t) for t in tier_opts_vals]
        tier_filter = st.selectbox("Tier", tier_opts, key="db2_tier_filter")
    with col_f4:
        show_n = st.number_input(
            "Show top N", min_value=20, max_value=max(len(available), 20),
            value=min(100, len(available)), step=20, key="db2_show_n"
        )
    with col_f5:
        search = st.text_input("Search name", placeholder="e.g. Ohtani", key="db2_search")

    filtered = available[:]
    if pos_filter != "All":
        filtered = [p for p in filtered if any(pos_filter in pos for pos in p["positions"])]
    if type_filter != "All":
        filtered = [p for p in filtered if p["type"] == type_filter.lower()]
    if tier_filter != "All":
        filtered = [p for p in filtered if p.get("tier") == int(tier_filter)]
    if search and len(search) >= 2:
        filtered = [p for p in filtered if search.lower() in p["name"].lower()]
    filtered = sorted(filtered, key=lambda p: p.get("z_score", 0), reverse=True)
    filtered = filtered[:int(show_n)]

    def _adp_str(adp):
        return f"{adp:.0f}" if adp < 999 else "UD"

    board_rows = []
    for p in filtered:
        proj = p.get("proj", {})
        board_rows.append({
            "Player": p["name"],
            "Pos": "/".join(p["positions"][:3]),
            "ADP": _adp_str(p.get("adp", 999)),
            "Z": round(p.get("z_score", 0), 2),
            "Tier": p.get("tier", "-"),
            "HR": int(proj.get("hr", 0)),
            "RBI": int(proj.get("rbi", 0)),
            "R": int(proj.get("r", 0)),
            "SB": int(proj.get("nsb", proj.get("sb", 0))),
            "AVG": round(proj.get("avg", 0), 3) if p["type"] == "batter" else "-",
            "ERA": round(proj.get("era", 0), 2) if p["type"] == "pitcher" else "-",
            "WHIP": round(proj.get("whip", 0), 2) if p["type"] == "pitcher" else "-",
            "K": int(proj.get("k_pit", proj.get("k_bat", 0))),
            "W": int(proj.get("w", 0)) if p["type"] == "pitcher" else "-",
            "SV": int(proj.get("sv", proj.get("nsv", 0))) if p["type"] == "pitcher" else "-",
        })

    board_df = pd.DataFrame(board_rows)

    def _tier_bg(val):
        colors = {1: "#1a472a", 2: "#2e4057", 3: "#5e3a07", 4: "#4a1628", 5: "#2d3436"}
        try:
            return f"background-color: {colors.get(int(val), '#1e1e1e')}; color: white"
        except (ValueError, TypeError):
            return ""

    st.dataframe(
        board_df.style.map(_tier_bg, subset=["Tier"]),
        use_container_width=True,
        hide_index=True,
        height=480,
    )
    st.caption(f"Showing {len(filtered)} available players (sorted by Z-Score)")

    # -----------------------------------------------------------------------
    # Section 3: Draft a Player by Name
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Draft a Player")
    st.caption("Type a name to cross them off the board (your pick or an opponent's).")

    draft_col1, draft_col2 = st.columns([3, 1])
    with draft_col1:
        draft_name = st.text_input(
            "Player name", placeholder="e.g. Aaron Judge", key="draft_name_input"
        )
    with draft_col2:
        is_my_pick = st.checkbox("This is MY pick", value=True, key="is_my_pick")

    if draft_name and len(draft_name) >= 3:
        matches = [
            p for p in available
            if draft_name.lower() in p["name"].lower()
        ]
        if matches:
            match_names = [p["name"] for p in matches[:10]]
            selected_name = st.selectbox("Select player", match_names, key="draft_select")
            selected_player = next(p for p in matches if p["name"] == selected_name)
            if st.button("Confirm Draft", type="primary", key="confirm_draft_btn"):
                if selected_player["id"] not in drafted_ids:
                    entry = dict(selected_player)
                    entry["_my_pick"] = is_my_pick
                    st.session_state["drafted"].append(entry)
                    if is_my_pick:
                        st.session_state["draft_pick_num"] = int(my_pick) + 1
                    st.success(f"Drafted: {selected_player['name']}")
                    st.rerun()
        else:
            st.warning(f"No available player matching '{draft_name}'")

    # -----------------------------------------------------------------------
    # Section 4: My Drafted Team + Category Totals
    # -----------------------------------------------------------------------
    my_picks = [p for p in st.session_state["drafted"] if p.get("_my_pick", True)]

    if my_picks:
        st.markdown("---")
        st.subheader(f"My Team ({len(my_picks)}/{TOTAL_PICKS} picks)")

        # Category totals
        totals = {
            "HR": 0, "RBI": 0, "R": 0, "SB": 0,
            "W": 0, "K": 0, "SV": 0,
            "_avg_sum": 0.0, "_avg_pa": 0,
            "_era_ip": 0.0, "_era_er": 0.0,
            "_whip_ip": 0.0, "_whip_bw": 0.0,
        }
        team_rows = []
        for p in my_picks:
            proj = p.get("proj", {})
            if p["type"] == "batter":
                totals["HR"] += int(proj.get("hr", 0))
                totals["RBI"] += int(proj.get("rbi", 0))
                totals["R"] += int(proj.get("r", 0))
                totals["SB"] += int(proj.get("nsb", proj.get("sb", 0)))
                pa = proj.get("pa", 0)
                avg = proj.get("avg", 0)
                totals["_avg_sum"] += avg * pa
                totals["_avg_pa"] += pa
            else:
                ip = proj.get("ip", 0)
                totals["W"] += int(proj.get("w", 0))
                totals["K"] += int(proj.get("k_pit", 0))
                totals["SV"] += int(proj.get("sv", proj.get("nsv", 0)))
                era = proj.get("era", 4.50)
                whip = proj.get("whip", 1.30)
                totals["_era_ip"] += ip
                totals["_era_er"] += era * ip / 9.0
                totals["_whip_ip"] += ip
                totals["_whip_bw"] += whip * ip

            team_rows.append({
                "Player": p["name"],
                "Pos": "/".join(p["positions"][:3]),
                "Type": p["type"].title(),
                "Z": round(p.get("z_score", 0), 2),
            })

        team_avg = (
            totals["_avg_sum"] / totals["_avg_pa"]
            if totals["_avg_pa"] > 0 else 0.0
        )
        team_era = (
            totals["_era_er"] / totals["_era_ip"] * 9
            if totals["_era_ip"] > 0 else 0.0
        )
        team_whip = (
            totals["_whip_bw"] / totals["_whip_ip"]
            if totals["_whip_ip"] > 0 else 0.0
        )

        # Display category summary
        cat_cols = st.columns(10)
        cat_cols[0].metric("HR", totals["HR"])
        cat_cols[1].metric("RBI", totals["RBI"])
        cat_cols[2].metric("R", totals["R"])
        cat_cols[3].metric("SB", totals["SB"])
        cat_cols[4].metric("AVG", f"{team_avg:.3f}")
        cat_cols[5].metric("ERA", f"{team_era:.2f}")
        cat_cols[6].metric("WHIP", f"{team_whip:.2f}")
        cat_cols[7].metric("K", totals["K"])
        cat_cols[8].metric("W", totals["W"])
        cat_cols[9].metric("SV", totals["SV"])

        st.dataframe(pd.DataFrame(team_rows), use_container_width=True, hide_index=True)

        if st.button("Undo Last Pick", key="undo_pick_btn"):
            if st.session_state["drafted"]:
                removed = st.session_state["drafted"].pop()
                if removed.get("_my_pick", True):
                    st.session_state["draft_pick_num"] = max(1, int(my_pick) - 1)
                st.rerun()

        if st.button("Reset Draft Board", key="reset_draft_btn"):
            st.session_state["drafted"] = []
            st.session_state["draft_pick_num"] = 1
            st.rerun()

    # -----------------------------------------------------------------------
    # Section 5: Drafted Players Full Log
    # -----------------------------------------------------------------------
    all_drafted = st.session_state["drafted"]
    if all_drafted:
        with st.expander(f"All Drafted Players ({len(all_drafted)} total — including opponents)"):
            drafted_rows = []
            for i, p in enumerate(all_drafted, 1):
                drafted_rows.append({
                    "Pick": i,
                    "Player": p["name"],
                    "Pos": "/".join(p["positions"][:3]),
                    "Type": p["type"].title(),
                    "Mine": "YES" if p.get("_my_pick", True) else "opp",
                })
            st.dataframe(pd.DataFrame(drafted_rows), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Section 6: Position Scarcity + Strategy
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Position Scarcity Snapshot")
    scarcity_cols = st.columns(5)
    pos_groups_sc = {
        "C": ["C"], "SS": ["SS"], "2B": ["2B"],
        "SP Top 10": ["SP"], "RP/Closer": ["RP"],
    }
    for col, (label, pos_list) in zip(scarcity_cols, pos_groups_sc.items()):
        t1_t2 = [
            p for p in available
            if p.get("tier", 99) <= 2 and any(pos in p["positions"] for pos in pos_list)
        ]
        col.metric(label, f"{len(t1_t2)} Tier 1-2 left")

    with st.expander("Draft Strategy Framework (Treemendous)"):
        st.markdown("""
| Round | Target | Rationale |
|-------|--------|-----------|
| 1-3   | Elite multi-cat hitters | HR + R + RBI + TB + AVG/OPS |
| 4-6   | Ace SPs | W + K + K/9 + QS + ERA + WHIP |
| 7-9   | SB assets | NSB is scarce — stock up mid-rounds |
| 10-13 | Solid SP depth | Minimize L exposure |
| 14-17 | Closer pipeline | NSV very scarce; target 2-3 |
| 18-23 | Streaming SPs, bench | High K/9, low L risk |

**Key rule:** High-K sluggers (30%+ K) are double-penalized — K hurts batting AND drags AVG/OPS.
Target contact-power bats (Alvarez, Freeman, Goldschmidt types).
""")

# ===========================================================================
# TAB: My Roster
# ===========================================================================
with tab_roster:
    st.header("My Roster")
    yahoo_ready = bool(os.getenv("YAHOO_REFRESH_TOKEN"))
    if not yahoo_ready:
        st.warning("Yahoo API not authenticated yet. Go to **Setup & Auth** tab to connect.")
    else:
        st.info("Click below to sync your current roster from Yahoo.")
        if st.button("Sync Roster from Yahoo"):
            try:
                from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
                client = YahooFantasyClient()
                with st.spinner("Fetching roster..."):
                    roster = client.get_roster()
                if not roster:
                    st.info(
                        "No players on roster yet — this is expected before the draft. "
                        "**Draft day: March 23 @ 7:30am EDT.** Come back after the draft to see your team."
                    )
                else:
                    st.success(f"Fetched {len(roster)} players")
                    cols_present = [c for c in ["name", "team", "positions", "status"] if c in pd.DataFrame(roster).columns]
                    st.dataframe(pd.DataFrame(roster)[cols_present], use_container_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")

# ===========================================================================
# TAB: Waiver Wire
# ===========================================================================
with tab_waiver:
    st.header("Waiver Wire Intelligence")
    st.info("In-season feature — available after March 24 draft.")
    st.markdown("""
**Waiver Rules — Treemendous**
- Type: Continual rolling list
- Wait time: 1 day
- Max acquisitions/week: **8** (hard cap — prioritize wisely)
- Injured players cannot go directly to IL slot from waivers

**What the waiver engine will do:**
- Rank free agents by: (a) category need vs. opponent, (b) rest-of-season value
- Highlight streaming SPs with good matchups this week
- Track your waiver priority position
- Alert when a valuable player is dropped by another team
""")

# ===========================================================================
# TAB: Trade Analyzer
# ===========================================================================
with tab_trade:
    st.header("Trade Analyzer")
    st.info("In-season feature — trade deadline August 6, 2026.")
    st.markdown("""
**How it works:**
1. Enter the proposed trade (players you give, players you receive)
2. Engine computes net category delta across all 18 categories
3. Shows which categories improve, which weaken
4. Toggle: Win-now mode vs. rebuild mode
5. ROS (rest-of-season) projections used for all valuations

**Trade Deadline: August 6, 2026**
""")

# ===========================================================================
# TAB: Setup & Auth
# ===========================================================================
with tab_setup:
    st.header("Yahoo API Setup")

    yahoo_client_id = os.getenv("YAHOO_CLIENT_ID", "")
    yahoo_client_secret = os.getenv("YAHOO_CLIENT_SECRET", "")
    yahoo_refresh = os.getenv("YAHOO_REFRESH_TOKEN", "")

    # ---- Status indicators ----
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if yahoo_client_id:
            st.success(f"Client ID: {yahoo_client_id[:12]}...")
        else:
            st.warning("CLIENT_ID not in local env")
    with col_s2:
        if yahoo_client_secret:
            st.success("Client Secret: set")
        else:
            st.warning("CLIENT_SECRET not in local env")
    with col_s3:
        if yahoo_refresh:
            st.success("Refresh Token: ready")
        else:
            st.warning("Refresh Token: MISSING")

    st.markdown("---")

    # If env vars aren't available locally, allow manual entry so the OAuth
    # flow can still run (Railway vars are only visible on deployed instances).
    if not yahoo_client_id or not yahoo_client_secret:
        st.info(
            "YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET are set in Railway but not visible "
            "in this local environment. Enter them below to proceed with the OAuth flow — "
            "they are never stored by this form."
        )
        manual_id = st.text_input(
            "YAHOO_CLIENT_ID",
            type="password",
            placeholder="Paste your Yahoo App Client ID",
            key="manual_yahoo_client_id",
        )
        manual_secret = st.text_input(
            "YAHOO_CLIENT_SECRET",
            type="password",
            placeholder="Paste your Yahoo App Client Secret",
            key="manual_yahoo_client_secret",
        )
        if manual_id:
            yahoo_client_id = manual_id
        if manual_secret:
            yahoo_client_secret = manual_secret

    if yahoo_client_id and yahoo_client_secret:
        # ---- Step 1: Generate auth URL ----
        st.subheader("Step 1 — Authorize Yahoo (one-time)")
        st.info(
            "CLIENT_ID and SECRET are set. You just need to complete the one-time OAuth flow "
            "to get a **Refresh Token**, then add it to Railway."
        )

        if st.button("Generate Yahoo Authorization URL"):
            try:
                # Inject manually-entered creds into env for this call
                os.environ["YAHOO_CLIENT_ID"] = yahoo_client_id
                os.environ["YAHOO_CLIENT_SECRET"] = yahoo_client_secret
                from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
                client = YahooFantasyClient()
                auth_url = client.get_authorization_url()
                st.session_state["yahoo_auth_url"] = auth_url
            except Exception as e:
                st.error(f"Failed to generate URL: {e}")

        if "yahoo_auth_url" in st.session_state:
            st.markdown(f"**[Click here to authorize with Yahoo]({st.session_state['yahoo_auth_url']})**")
            st.caption("Yahoo will show you a 6-digit code after you approve. Copy it below.")

        # ---- Step 2: Exchange code for tokens ----
        st.subheader("Step 2 — Enter Authorization Code")
        auth_code_input = st.text_input(
            "Paste the 6-digit Yahoo authorization code here:",
            placeholder="e.g., abc123",
            key="yahoo_auth_code",
        )
        if st.button("Exchange Code for Tokens") and auth_code_input.strip():
            try:
                os.environ["YAHOO_CLIENT_ID"] = yahoo_client_id
                os.environ["YAHOO_CLIENT_SECRET"] = yahoo_client_secret
                from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
                client = YahooFantasyClient()
                tokens = client.exchange_code_for_tokens(auth_code_input.strip())
                st.session_state["yahoo_new_tokens"] = tokens
                st.success("Authorization successful! Copy the tokens below to Railway.")
            except Exception as e:
                st.error(f"Token exchange failed: {e}")

        if "yahoo_new_tokens" in st.session_state:
            t = st.session_state["yahoo_new_tokens"]
            st.subheader("Step 3 — Add These to Railway Environment Variables")
            st.warning("Copy these values to Railway → Variables before the page refreshes.")
            st.code(f"YAHOO_REFRESH_TOKEN={t.get('refresh_token', 'N/A')}")
            st.code(f"YAHOO_ACCESS_TOKEN={t.get('access_token', 'N/A')[:40]}...")
            st.markdown(
                "**Railway Dashboard → Your Project → Variables → Add** each key above. "
                "Then redeploy. The app will auto-refresh the access token on every restart."
            )

    st.markdown("---")

    # ---- Connection test ----
    if yahoo_refresh:
        st.subheader("Test Connection")
        col_t1, col_t2 = st.columns(2)
        if col_t1.button("Test Yahoo API Connection"):
            try:
                os.environ["YAHOO_CLIENT_ID"] = yahoo_client_id
                os.environ["YAHOO_CLIENT_SECRET"] = yahoo_client_secret
                from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
                client = YahooFantasyClient()
                league = client.get_league()
                st.success(f"Connected! League: **{league.get('name', 'Unknown')}**")
                my_key = client.get_my_team_key()
                st.info(f"Your team key: `{my_key}`")
            except Exception as e:
                st.error(f"Connection test failed: {e}")

        if col_t2.button("Debug: Show Raw Roster Response"):
            try:
                os.environ["YAHOO_CLIENT_ID"] = yahoo_client_id
                os.environ["YAHOO_CLIENT_SECRET"] = yahoo_client_secret
                from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
                client = YahooFantasyClient()
                raw = client.get_roster_raw()
                st.subheader("Raw `fantasy_content` from Yahoo")
                st.caption("This shows the exact structure Yahoo returned — helps diagnose parsing issues.")
                st.json(raw)
            except Exception as e:
                st.error(f"Debug failed: {e}")

    st.markdown("---")
    st.subheader("Troubleshooting")
    st.markdown("""
| Symptom | Cause | Fix |
|---------|-------|-----|
| "No refresh token stored" | `YAHOO_REFRESH_TOKEN` not in Railway | Complete OAuth flow above, copy token to Railway |
| "Token refresh failed: 400" | Refresh token expired or revoked | Re-do OAuth flow above |
| "Token exchange failed: 401" | Wrong CLIENT_ID or CLIENT_SECRET | Check Railway vars match developer.yahoo.com/apps |
| API returns empty data | Wrong league ID | Set `YAHOO_LEAGUE_ID=72586` in Railway |
| Works locally, fails on Railway | `.env` not writable | Expected — add tokens to Railway Variables directly |
""")


# ---------------------------------------------------------------------------
# Helper: baseline player pool for z-score context
# ---------------------------------------------------------------------------

def _build_baseline_pool() -> list[PlayerProjection]:
    """
    Approximate 2026 baseline player pool for z-score calibration.
    These represent typical drafted starters in a 12-team league.
    Weights are tuned for Treemendous category set.
    """
    batters = [
        # name, team, pa, r, h, hr, rbi, k_bat, tb, avg, ops, nsb
        ("Ronald Acuna Jr.", "ATL", 680, 115, 185, 35, 95, 155, 340, 0.302, 0.975, 55),
        ("Freddie Freeman", "LAD", 660, 100, 190, 28, 100, 120, 310, 0.300, 0.910, 10),
        ("Yordan Alvarez", "HOU", 620, 95, 165, 37, 108, 140, 340, 0.295, 0.970, 3),
        ("Mookie Betts", "LAD", 650, 110, 180, 30, 90, 135, 315, 0.295, 0.930, 15),
        ("Julio Rodriguez", "SEA", 650, 95, 175, 30, 90, 160, 300, 0.285, 0.870, 28),
        ("Corey Seager", "TEX", 580, 90, 165, 30, 95, 110, 295, 0.295, 0.920, 5),
        ("José Ramirez", "CLE", 650, 100, 175, 28, 100, 85, 300, 0.285, 0.880, 25),
        ("Elly De La Cruz", "CIN", 620, 95, 160, 25, 85, 200, 275, 0.265, 0.830, 50),
        ("Gunnar Henderson", "BAL", 640, 100, 170, 32, 95, 165, 310, 0.278, 0.890, 18),
        ("Juan Soto", "NYY", 680, 105, 172, 38, 100, 128, 330, 0.288, 0.977, 7),
        ("Bobby Witt Jr.", "KC", 660, 105, 185, 28, 95, 130, 300, 0.295, 0.890, 40),
        ("Adley Rutschman", "BAL", 580, 80, 155, 18, 80, 95, 255, 0.275, 0.840, 5),
        ("Trea Turner", "PHI", 620, 95, 175, 22, 80, 130, 285, 0.295, 0.860, 28),
        ("Francisco Lindor", "NYM", 640, 95, 165, 28, 95, 145, 295, 0.275, 0.860, 22),
        ("Marcus Semien", "TEX", 660, 100, 170, 25, 88, 135, 285, 0.270, 0.840, 12),
        ("William Contreras", "MIL", 560, 75, 150, 22, 80, 105, 250, 0.275, 0.840, 5),
        ("Bryce Harper", "PHI", 580, 90, 160, 30, 95, 120, 295, 0.285, 0.930, 8),
        ("Pete Alonso", "NYM", 620, 80, 150, 38, 105, 155, 305, 0.255, 0.870, 2),
        ("Spencer Steer", "CIN", 580, 78, 148, 22, 82, 130, 252, 0.265, 0.800, 12),
        ("Jazz Chisholm", "NYY", 560, 82, 145, 22, 78, 155, 250, 0.262, 0.810, 22),
    ]
    pitchers = [
        # name, team, ip, w, l, sv, bs, qs, k, era, whip, k9, hr, nsv
        ("Gerrit Cole", "NYY", 185, 14, 6, 0, 0, 22, 225, 2.80, 1.00, 10.9, 18, 0),
        ("Spencer Strider", "ATL", 175, 14, 5, 0, 0, 21, 240, 2.70, 0.95, 12.3, 16, 0),
        ("Logan Webb", "SF", 200, 14, 7, 0, 0, 24, 195, 3.10, 1.08, 8.8, 17, 0),
        ("Zack Wheeler", "PHI", 195, 14, 6, 0, 0, 24, 215, 3.00, 1.05, 9.9, 19, 0),
        ("Paul Skenes", "PIT", 170, 12, 7, 0, 0, 20, 220, 2.90, 1.02, 11.6, 16, 0),
        ("Corbin Burnes", "BAL", 190, 13, 7, 0, 0, 22, 205, 3.10, 1.06, 9.7, 18, 0),
        ("Emmanuel Clase", "CLE", 70, 4, 2, 38, 5, 0, 82, 2.20, 0.90, 10.5, 5, 33),
        ("Josh Hader", "HOU", 65, 3, 3, 35, 6, 0, 88, 2.40, 0.95, 12.2, 4, 29),
        ("Edwin Diaz", "NYM", 65, 3, 3, 32, 5, 0, 92, 2.50, 0.95, 12.7, 4, 27),
        ("Taijuan Walker", "PHI", 160, 10, 8, 0, 0, 18, 150, 3.80, 1.22, 8.4, 20, 0),
        ("Dylan Cease", "SD", 180, 12, 8, 0, 0, 20, 200, 3.40, 1.15, 10.0, 19, 0),
        ("Framber Valdez", "HOU", 195, 13, 8, 0, 0, 23, 175, 3.25, 1.18, 8.1, 16, 0),
        ("Kevin Gausman", "TOR", 185, 12, 8, 0, 0, 22, 195, 3.20, 1.05, 9.5, 20, 0),
        ("Shane Bieber", "CLE", 175, 12, 7, 0, 0, 21, 190, 3.15, 1.05, 9.8, 18, 0),
        ("Camilo Doval", "SF", 65, 3, 4, 30, 7, 0, 72, 2.80, 1.05, 9.9, 5, 23),
    ]

    pool = []
    for b in batters:
        (name, team, pa, r, h, hr, rbi, k_bat, tb, avg, ops, nsb) = b
        p = PlayerProjection(
            name=name, yahoo_player_key="", team=team,
            positions=["OF"], player_type="batter",
            pa=pa, r=r, h=h, hr=hr, rbi=rbi, k_bat=k_bat,
            tb=tb, avg=avg, ops=ops, nsb=nsb,
        )
        p.slg = ops - 0.330
        pool.append(p)
    for pi in pitchers:
        (name, team, ip, w, l, sv, bs, qs, k, era, whip, k9, hr_p, nsv) = pi
        p = PlayerProjection(
            name=name, yahoo_player_key="", team=team,
            positions=["SP"] if sv == 0 else ["RP"], player_type="pitcher",
            ip=ip, w=w, l=l, sv=sv, bs=bs, qs=qs, k_pit=k,
            era=era, whip=whip, k9=k9, hr_pit=hr_p, nsv=nsv,
        )
        pool.append(p)
    return pool
