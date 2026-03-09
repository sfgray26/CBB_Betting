"""
Live Draft Assistant — Treemendous League
90-second clock · 12 teams · Rounds 1-2 linear, then snake
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.fantasy_baseball.draft_engine import (
    DraftState, DraftRecommender, NUM_TEAMS, NUM_ROUNDS,
    build_full_pick_order, picks_for_position, get_pick_order,
)
from backend.fantasy_baseball.player_board import get_board, get_player_by_name, available_players
from backend.fantasy_baseball.qwen_advisor import draft_pick_rationale, is_ollama_available

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Live Draft Assistant",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
if "draft_state" not in st.session_state:
    st.session_state.draft_state = None
if "board" not in st.session_state:
    st.session_state.board = get_board()
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []
if "pick_log" not in st.session_state:
    st.session_state.pick_log = []
if "draft_started" not in st.session_state:
    st.session_state.draft_started = False
if "qwen_rationales" not in st.session_state:
    st.session_state.qwen_rationales = {}

board = st.session_state.board

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚾ Live Draft Assistant — Treemendous")

qwen_status = "🟢 Qwen online" if is_ollama_available() else "🔴 Qwen offline (using pre-computed rationale)"
st.caption(
    f"12 teams · H2H One Win · 18 cats · Rounds 1–2 linear, then snake · "
    f"90-sec clock · {qwen_status}"
)

# ---------------------------------------------------------------------------
# SIDEBAR — Draft setup & status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Draft Setup")

    if not st.session_state.draft_started:
        draft_pos = st.number_input(
            "Your draft position (1–12)",
            min_value=1, max_value=12, value=7,
            help="Position 12 gets picks 12, 24, 25 — three consecutive picks!"
        )
        st.markdown("---")
        st.markdown("**Pick order reminder:**")
        st.markdown("- Round 1: 1→12 (linear)")
        st.markdown("- Round 2: 1→12 (same order)")
        st.markdown("- Round 3+: Snake (reverses)")

        # Preview pick schedule for selected position
        my_picks = picks_for_position(draft_pos)
        preview_df = pd.DataFrame(
            [(pk, rnd) for pk, rnd in my_picks[:8]],
            columns=["Overall Pick", "Round"]
        )
        st.markdown(f"**Your picks (position {draft_pos}):**")
        st.dataframe(preview_df, hide_index=True, use_container_width=True)

        if st.button("🚀 Start Draft", type="primary", use_container_width=True):
            st.session_state.draft_state = DraftState(
                my_draft_position=draft_pos,
                num_teams=NUM_TEAMS,
                num_rounds=NUM_ROUNDS,
            )
            st.session_state.draft_started = True
            st.rerun()
    else:
        state: DraftState = st.session_state.draft_state
        st.markdown(f"**My position:** {state.my_draft_position}")
        st.markdown(f"**Picks made:** {len(state.picks_made)}")
        st.markdown(f"**My roster:** {len(state.my_roster)} players")

        nxt = state.next_my_pick()
        if nxt:
            picks_away = state.picks_until_my_turn()
            if picks_away == 0:
                st.error("🔴 YOUR PICK NOW!")
            elif picks_away <= 3:
                st.warning(f"⚠️ {picks_away} picks until your turn")
            else:
                st.info(f"⏳ {picks_away} picks until your turn (Pick {nxt[0]}, Round {nxt[1]})")

        st.markdown("---")
        st.markdown("**My Roster Needs:**")
        needs = state.roster_needs()
        for slot, count in sorted(needs.items()):
            urgency = "🔴" if slot in ("C", "SS") else "🟡"
            st.markdown(f"{urgency} {slot}: {count} needed")

        if st.button("Reset Draft", type="secondary", use_container_width=True):
            st.session_state.draft_started = False
            st.session_state.draft_state = None
            st.session_state.last_recs = []
            st.session_state.pick_log = []
            st.session_state.qwen_rationales = {}
            st.rerun()

# ---------------------------------------------------------------------------
# Pre-draft view (setup not complete)
# ---------------------------------------------------------------------------
if not st.session_state.draft_started:
    st.info("Set your draft position in the sidebar and click **Start Draft** to begin.")

    # Show full board preview
    st.subheader("Full Player Board Preview")
    df = pd.DataFrame([{
        "Rank": p["rank"], "Tier": p["tier"], "Name": p["name"],
        "Team": p["team"], "Positions": "/".join(p["positions"][:3]),
        "Type": p["type"].title(), "ADP": p["adp"],
        "Z-Score": f"{p['z_score']:+.2f}",
    } for p in board[:80]])

    pos_filter = st.selectbox("Filter by position", ["All", "SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "DH"])
    type_filter = st.selectbox("Filter by type", ["All", "Batter", "Pitcher"])

    filtered = board[:]
    if pos_filter != "All":
        filtered = [p for p in filtered if any(pos_filter in pos for pos in p["positions"])]
    if type_filter != "All":
        filtered = [p for p in filtered if p["type"] == type_filter.lower()]

    df_filtered = pd.DataFrame([{
        "Rank": p["rank"], "Tier": p["tier"], "Name": p["name"],
        "Team": p["team"], "Positions": "/".join(p["positions"][:3]),
        "Type": p["type"].title(), "ADP": p["adp"],
        "Z-Score": f"{p['z_score']:+.2f}",
    } for p in filtered[:100]])

    def tier_color(val):
        colors = {1: "#1a472a", 2: "#2e4057", 3: "#5e3a07",
                  4: "#4a1628", 5: "#2d3436", 6: "#2d3436"}
        return f"background-color: {colors.get(val, '#1e1e1e')}; color: white"

    st.dataframe(
        df_filtered.style.applymap(tier_color, subset=["Tier"]),
        use_container_width=True,
        hide_index=True,
        height=500,
    )
    st.stop()

# ---------------------------------------------------------------------------
# LIVE DRAFT VIEW
# ---------------------------------------------------------------------------
state: DraftState = st.session_state.draft_state
avail = available_players(state.drafted_player_ids)
recommender = DraftRecommender(state, avail)

# -- Status bar --------------------------------------------------------------
picks_away = state.picks_until_my_turn()
col_pick, col_round, col_pos, col_turn = st.columns(4)
col_pick.metric("Overall Pick", state.current_overall_pick)
col_round.metric("Round", state.current_round)
col_pos.metric("On the Clock", f"Position {state.current_draft_position}")
if state.is_my_pick:
    col_turn.error("🔴 YOUR PICK!")
elif picks_away <= 2:
    col_turn.warning(f"⚠️ {picks_away} picks away")
else:
    nxt = state.next_my_pick()
    col_turn.info(f"Next: Pick {nxt[0]} R{nxt[1]}" if nxt else "Draft over")

st.divider()

# -- Main layout: recommendations | my roster | pick entry ------------------
col_left, col_right = st.columns([3, 2])

# -- LEFT: Recommendations ---------------------------------------------------
with col_left:
    rec_tab, board_tab = st.tabs(["🎯 Recommendations", "📋 Full Board"])

    with rec_tab:
        if state.is_my_pick:
            st.subheader("🔴 YOUR PICK — Top Recommendations")
        else:
            st.subheader(f"Top Picks (ready when you're up in {picks_away} picks)")

        # -- Look-ahead intelligence (when not my turn) ----------------------
        if not state.is_my_pick:
            la = recommender.look_ahead()
            if la.get("likely_gone"):
                with st.expander(
                    f"⚠️ {len(la['likely_gone'])} players likely gone before your pick {la.get('my_next_pick', '?')} (Round {la.get('my_next_round', '?')})",
                    expanded=(picks_away <= 5)
                ):
                    gone_df = pd.DataFrame([{
                        "Name": p["name"], "Pos": "/".join(p["positions"][:2]),
                        "Team": p["team"], "ADP": f"{p['adp']:.0f}",
                        "Tier": p["tier"], "Z": f"{p['z_score']:+.2f}",
                    } for p in la["likely_gone"]])
                    st.dataframe(gone_df, hide_index=True, use_container_width=True)

                if la.get("top_targets_at_my_pick"):
                    st.markdown("**Best available at your pick:**")
                    for p in la["top_targets_at_my_pick"][:3]:
                        pos = "/".join(p["positions"][:2])
                        risk = p.get("injury_risk", "")
                        risk_tag = " ⚠️" if risk in ("high", "extreme") else ""
                        st.markdown(f"  → **{p['name']}** ({pos}, {p['team']}) T{p['tier']} Z={p.get('z_score', 0):+.2f}{risk_tag}")

                if la.get("potential_sleepers"):
                    st.markdown("**Potential sleepers (ADP late but good value):**")
                    for p in la["potential_sleepers"]:
                        pos = "/".join(p["positions"][:2])
                        st.markdown(f"  → **{p['name']}** ({pos}) ADP {p['adp']:.0f} vs Z={p.get('z_score', 0):+.2f}")

        # Get recs
        recs = recommender.recommend(top_n=8)
        st.session_state.last_recs = recs

        for i, rec in enumerate(recs[:5], 1):
            pos_str = "/".join(rec.positions[:3])
            adp_diff_str = f"+{rec.adp_diff:.0f}" if rec.adp_diff > 0 else f"{rec.adp_diff:.0f}"
            tier_badge = f"T{rec.tier}"

            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    reach_tag = " ⚠️REACH" if rec.reach_alert else ""
                    value_tag = " 🟢VALUE" if rec.adp_diff > 10 else ""
                    st.markdown(
                        f"**{i}. {rec.player_name}**{reach_tag}{value_tag}  \n"
                        f"`{tier_badge}` · {pos_str} · {rec.team} · ADP {rec.adp:.0f} ({adp_diff_str})"
                    )
                    # Rationale — use Qwen if available, else pre-computed
                    rationale_key = rec.player_id
                    if rationale_key not in st.session_state.qwen_rationales:
                        rationale = rec.rationale
                        if is_ollama_available() and state.is_my_pick:
                            with st.spinner("Qwen thinking..."):
                                rationale = draft_pick_rationale(
                                    rec.player_name, rec.positions, rec.player_type,
                                    state.current_round, state.roster_needs(),
                                    rec.top_categories,
                                )
                            st.session_state.qwen_rationales[rationale_key] = rationale
                        else:
                            st.session_state.qwen_rationales[rationale_key] = rationale
                    st.caption(st.session_state.qwen_rationales.get(rationale_key, rec.rationale))
                with c2:
                    st.metric("Z-Score", f"{rec.z_score:+.2f}")
                with c3:
                    st.metric("Need×", f"{rec.need_boost:.1f}x")

                # Quick pick button
                if st.button(f"✓ Draft {rec.player_name}", key=f"draft_rec_{i}_{rec.player_id}"):
                    state.log_pick(
                        rec.player_id, rec.player_name, rec.team,
                        rec.positions, rec.player_type,
                    )
                    st.session_state.pick_log.append({
                        "pick": state.current_overall_pick - 1,
                        "round": state.current_round,
                        "who": f"You (pos {state.my_draft_position})",
                        "player": rec.player_name,
                        "positions": "/".join(rec.positions[:2]),
                    })
                    st.rerun()
                st.markdown("---")

    with board_tab:
        st.subheader("Available Players")
        pos_filter2 = st.selectbox("Position filter", ["All", "SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "DH"], key="board_pos_filter")
        type_filter2 = st.selectbox("Type filter", ["All", "Batter", "Pitcher"], key="board_type_filter")

        filtered_avail = avail[:]
        if pos_filter2 != "All":
            filtered_avail = [p for p in filtered_avail if any(pos_filter2 in pos for pos in p["positions"])]
        if type_filter2 != "All":
            filtered_avail = [p for p in filtered_avail if p["type"] == type_filter2.lower()]

        board_df = pd.DataFrame([{
            "Rank": p["rank"], "T": p["tier"], "Name": p["name"],
            "Team": p["team"], "Pos": "/".join(p["positions"][:3]),
            "ADP": f"{p['adp']:.0f}",
            "Z": f"{p['z_score']:+.2f}",
            "Z(park)": f"{p.get('z_park_adjusted', p.get('z_score', 0)):+.2f}",
            "Risk": p.get("injury_risk", "").upper()[:3] if p.get("injury_risk", "") not in ("", "low") else "",
        } for p in filtered_avail[:120]])

        # Make it clickable — user selects a row to draft
        selection = st.dataframe(
            board_df,
            use_container_width=True,
            hide_index=True,
            height=500,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Draft selected player
        if selection and selection.selection.rows:
            row_idx = selection.selection.rows[0]
            selected_player = filtered_avail[row_idx]
            st.info(f"Selected: **{selected_player['name']}** — {'/'.join(selected_player['positions'][:2])}")

            whose_pick = st.radio(
                "Who picked this player?",
                [f"Me (position {state.my_draft_position})", "Another team"],
                horizontal=True,
                key="whose_pick_board",
            )
            if st.button(f"✓ Confirm: {selected_player['name']}", type="primary", key="confirm_board_pick"):
                is_mine = "Me" in whose_pick
                state.log_pick(
                    selected_player["id"], selected_player["name"],
                    selected_player["team"], selected_player["positions"],
                    selected_player["type"],
                )
                st.session_state.pick_log.append({
                    "pick": state.current_overall_pick - 1,
                    "round": state.current_round,
                    "who": "You" if is_mine else f"Pos {state.current_draft_position}",
                    "player": selected_player["name"],
                    "positions": "/".join(selected_player["positions"][:2]),
                })
                st.rerun()

# -- RIGHT: Roster + Pick entry ----------------------------------------------
with col_right:
    roster_tab, manual_tab, log_tab = st.tabs(["👥 My Roster", "⌨️ Log a Pick", "📜 Pick Log"])

    with roster_tab:
        st.subheader("My Roster")
        if not state.my_roster:
            st.caption("No picks yet — waiting for your first pick.")
        else:
            for p in state.my_roster:
                pos_str = "/".join(p.positions[:2])
                st.markdown(f"**R{p.round_num}** — {p.player_name} ({p.team}, {pos_str})")

        st.markdown("---")
        st.subheader("Category Balance")
        st.caption("Estimates based on drafted players' projections")

        # Build a quick category summary from drafted players
        drafted_names = [p.player_name for p in state.my_roster]
        if drafted_names:
            my_players = [p for p in board if p["name"] in drafted_names]
            bat_players = [p for p in my_players if p["type"] == "batter"]
            pit_players = [p for p in my_players if p["type"] == "pitcher"]

            bat_totals = {
                "R": sum(p["proj"].get("r", 0) for p in bat_players),
                "HR": sum(p["proj"].get("hr", 0) for p in bat_players),
                "NSB": sum(p["proj"].get("nsb", 0) for p in bat_players),
                "OPS": (sum(p["proj"].get("ops", 0) * p["proj"].get("pa", 1) for p in bat_players)
                        / max(sum(p["proj"].get("pa", 1) for p in bat_players), 1)),
            }
            pit_totals = {
                "W": sum(p["proj"].get("w", 0) for p in pit_players),
                "K": sum(p["proj"].get("k_pit", 0) for p in pit_players),
                "NSV": sum(p["proj"].get("nsv", 0) for p in pit_players),
                "ERA": (sum(p["proj"].get("era", 0) * p["proj"].get("ip", 0) for p in pit_players) * 9
                        / max(sum(p["proj"].get("ip", 0) for p in pit_players), 1)),
            }

            cat_df = pd.DataFrame([
                {"Category": "R", "Projected": f"{bat_totals['R']:.0f}", "Note": ""},
                {"Category": "HR", "Projected": f"{bat_totals['HR']:.0f}", "Note": ""},
                {"Category": "NSB", "Projected": f"{bat_totals['NSB']:.0f}", "Note": "🔑 Scarce" if bat_totals["NSB"] < 20 else ""},
                {"Category": "OPS", "Projected": f"{bat_totals['OPS']:.3f}", "Note": ""},
                {"Category": "W", "Projected": f"{pit_totals['W']:.0f}", "Note": ""},
                {"Category": "K (P)", "Projected": f"{pit_totals['K']:.0f}", "Note": ""},
                {"Category": "NSV", "Projected": f"{pit_totals['NSV']:.0f}", "Note": "🔑 Scarce" if pit_totals["NSV"] < 20 else ""},
                {"Category": "ERA", "Projected": f"{pit_totals['ERA']:.2f}", "Note": ""},
            ])
            st.dataframe(cat_df, hide_index=True, use_container_width=True)

    with manual_tab:
        st.subheader("Log Any Pick")
        st.caption("Use this to log other teams' picks to keep the board current.")

        search_name = st.text_input("Search player name", placeholder="e.g. Acuna")
        whose = st.radio(
            "Whose pick is this?",
            [f"Mine (pos {state.my_draft_position})", "Another team"],
            horizontal=True,
            key="manual_whose",
        )

        matched_player = None
        if search_name and len(search_name) >= 3:
            matches = [p for p in avail if search_name.lower() in p["name"].lower()]
            if matches:
                options = [f"{p['name']} ({p['team']}, {'/'.join(p['positions'][:2])})" for p in matches[:8]]
                selected_idx = st.selectbox("Select player", range(len(options)), format_func=lambda i: options[i])
                matched_player = matches[selected_idx]
                st.markdown(f"**ADP:** {matched_player['adp']:.0f} · **Tier:** {matched_player['tier']} · **Z-Score:** {matched_player['z_score']:+.2f}")

        if matched_player and st.button("✓ Log Pick", type="primary", key="manual_log"):
            state.log_pick(
                matched_player["id"], matched_player["name"],
                matched_player["team"], matched_player["positions"],
                matched_player["type"],
            )
            # If it's mine, manually override ownership
            if "Mine" in whose:
                # Already handled by DraftState (it checks position match)
                # but we'll mark it as mine if position doesn't match (manual override)
                if not state.my_roster or state.my_roster[-1].player_id != matched_player["id"]:
                    state.my_roster.append(state.picks_made[-1])

            st.session_state.pick_log.append({
                "pick": state.current_overall_pick - 1,
                "round": state.current_round,
                "who": "You" if "Mine" in whose else f"Pos {state.current_draft_position}",
                "player": matched_player["name"],
                "positions": "/".join(matched_player["positions"][:2]),
            })
            avail = available_players(state.drafted_player_ids)
            st.success(f"Logged: {matched_player['name']}")
            st.rerun()

    with log_tab:
        st.subheader("Pick Log")
        if st.session_state.pick_log:
            log_df = pd.DataFrame(st.session_state.pick_log)
            st.dataframe(log_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No picks logged yet.")

# ---------------------------------------------------------------------------
# Bottom: Upcoming picks timeline
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📅 Upcoming Picks Timeline")

full_order = build_full_pick_order()
current_pick = state.current_overall_pick
upcoming_picks = [
    (pk, rnd, pos) for pk, rnd, pos in full_order
    if pk >= current_pick and pk <= current_pick + 15
]
timeline_data = []
for pk, rnd, pos in upcoming_picks:
    is_mine = (pos == state.my_draft_position)
    timeline_data.append({
        "Pick #": pk, "Round": rnd, "Draft Position": pos,
        "Who": f">>> YOU <<<" if is_mine else f"Team {pos}",
    })
if timeline_data:
    timeline_df = pd.DataFrame(timeline_data)

    def highlight_mine(row):
        if "YOU" in str(row.get("Who", "")):
            return ["background-color: #1a472a; color: white"] * len(row)
        return [""] * len(row)

    st.dataframe(
        timeline_df.style.apply(highlight_mine, axis=1),
        use_container_width=True,
        hide_index=True,
    )
