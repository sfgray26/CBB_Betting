"""Admin Panel — manual job triggers, scheduler, portfolio gauges, config,
DraftKings import, bankroll override, and parlay force override."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from dashboard.utils import api_get, api_post, api_delete, sidebar_api_key, get_float_env
from dashboard.shared import inject_custom_css

st.set_page_config(page_title="Admin Panel | CBB Edge", layout="wide")
sidebar_api_key()
inject_custom_css()

st.title("Admin Panel")
st.caption("Operator controls — requires admin API key.")

# --- Manual job triggers ---
st.subheader("Manual Job Triggers")

col1, col2, col3 = st.columns(3)

with col1:
    notify = st.checkbox("Notify Discord after analysis", value=False)
    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running nightly analysis (this may take several minutes)..."):
            result = api_post(f"/admin/run-analysis?notify_discord={str(notify).lower()}", {}, timeout=300)
        if result:
            st.success(f"Analysis complete: {result.get('games_analyzed', '?')} games analyzed")
        else:
            st.error("Analysis failed — check API logs.")

with col2:
    if st.button("Force Update Outcomes", use_container_width=True):
        with st.spinner("Updating completed games..."):
            result = api_post("/admin/force-update-outcomes", {}, timeout=120)
        if result:
            st.success(f"Updated {result.get('games_updated', '?')} games")
        else:
            st.error("Update failed — check API logs.")

with col3:
    if st.button("Force Capture Lines", use_container_width=True):
        with st.spinner("Capturing closing lines..."):
            result = api_post("/admin/force-capture-lines", {}, timeout=120)
        if result:
            st.success(f"Captured lines for {result.get('games_captured', '?')} games")
        else:
            st.error("Capture failed — check API logs.")

st.markdown("---")

# ============================================================================
# DATA CLEANUP — delete stale/test game records
# ============================================================================
st.subheader("Data Cleanup")
st.caption("Delete a stale or test game and all its associated predictions. Blocked if real bet logs exist.")

with st.form("delete_game_form"):
    del_game_id = st.number_input("Game ID to delete", min_value=1, step=1, value=1)
    del_submitted = st.form_submit_button("Delete Game", type="secondary")

if del_submitted:
    result = api_delete(f"/admin/games/{int(del_game_id)}")
    if result:
        st.success(
            f"Deleted game {result['game_id']} ({result['matchup']}) — "
            f"{result['predictions_deleted']} prediction(s) removed."
        )

st.markdown("---")

# ============================================================================
# BANKROLL OVERRIDE
# ============================================================================
st.subheader("Bankroll Override")

bankroll_data = api_get("/admin/bankroll")
if bankroll_data:
    effective = bankroll_data.get("effective_bankroll", 0)
    source = bankroll_data.get("source", "unknown")
    env_val = bankroll_data.get("env_starting_bankroll", 0)
    last_set = bankroll_data.get("last_set", "—")

    b1, b2, b3 = st.columns(3)
    b1.metric("Effective Bankroll", f"${effective:,.2f}")
    b2.metric("Source", source.replace("_", " ").title())
    b3.metric("Env STARTING_BANKROLL", f"${env_val:,.0f}")
    if last_set and last_set != "—":
        st.caption(f"Last set: {last_set[:19].replace('T', ' ')} UTC")

    with st.form("bankroll_override_form"):
        new_bankroll = st.number_input(
            "Set new bankroll ($)",
            min_value=10.0,
            max_value=1_000_000.0,
            value=float(effective),
            step=10.0,
            format="%.2f",
            help="This overrides STARTING_BANKROLL for Kelly sizing in the next analysis run.",
        )
        submitted = st.form_submit_button("Apply Bankroll Override", type="primary")
        if submitted:
            result = api_post(f"/admin/bankroll?amount={new_bankroll}", {})
            if result and result.get("status") == "ok":
                st.success(f"Bankroll set to ${result['bankroll_set']:,.2f}")
                st.rerun()
            else:
                st.error("Failed to set bankroll.")
else:
    st.warning("Could not fetch bankroll data.")

st.markdown("---")

# ============================================================================
# PARLAY FORCE OVERRIDE
# ============================================================================
st.subheader("Parlay Force Override")

parlay_override = api_get("/admin/parlay/override")
if parlay_override:
    is_active = parlay_override.get("force_parlay_sizing", False)
    last_set_p = parlay_override.get("last_set", "—")

    status_color = "🟢" if is_active else "⚪"
    st.write(f"{status_color} Force parlay sizing is currently **{'ACTIVE' if is_active else 'INACTIVE'}**")
    if last_set_p and last_set_p != "—":
        st.caption(f"Last changed: {last_set_p[:19].replace('T', ' ')} UTC")

    p1, p2 = st.columns(2)
    with p1:
        if st.button("Enable Force Parlay", disabled=is_active, use_container_width=True):
            r = api_post("/admin/parlay/override?active=true", {})
            if r and r.get("status") == "ok":
                st.success("Force parlay sizing enabled.")
                st.rerun()
    with p2:
        if st.button("Disable Force Parlay", disabled=not is_active, use_container_width=True, type="secondary"):
            r = api_post("/admin/parlay/override?active=false", {})
            if r and r.get("status") == "ok":
                st.info("Force parlay sizing disabled.")
                st.rerun()

    st.caption(
        "When enabled, parlay recommendations are generated even when the daily straight-bet "
        "budget is exhausted. Parlays are sized at a minimum of 0.25u."
    )
else:
    st.warning("Could not fetch parlay override status.")

st.markdown("---")

# ============================================================================
# DRAFTKINGS CSV IMPORT
# ============================================================================
st.subheader("DraftKings CSV Import")
st.caption(
    "Upload your DraftKings transaction history CSV to import real bet results. "
    "Download from DraftKings → My Account → Transaction History → Export CSV."
)

import pandas as pd

dk_tab1, dk_tab2 = st.tabs(["Direct Import (Recommended)", "Match to Paper Trades"])

# --------------------------------------------------------------------------
# TAB 1: DIRECT IMPORT — creates new real BetLog entries from DK CSV
# --------------------------------------------------------------------------
with dk_tab1:
    st.write(
        "**Creates real bet records directly from your DK transaction history.** "
        "No model paper trades required — each DK wager is matched to the closest "
        "game in the database by tipoff time. Review the matched games below and "
        "correct any wrong Game IDs before confirming."
    )

    direct_file = st.file_uploader(
        "Upload DraftKings transaction history (.csv)",
        type=["csv"],
        key="dk_direct_uploader",
    )

    if direct_file is not None:
        direct_csv = direct_file.read().decode("utf-8", errors="replace")
        st.success(f"File loaded: {direct_file.name} ({len(direct_csv):,} bytes)")

        if st.button("Preview Direct Import", type="primary", key="dk_direct_preview_btn"):
            with st.spinner("Parsing CSV and matching wagers to games by tipoff time..."):
                preview = api_post(
                    "/admin/dk/direct-preview", {"csv_content": direct_csv}, timeout=60
                )

            if preview is None:
                st.error("Preview failed — check API logs.")
            else:
                n_wagers = preview.get("wagers_found", 0)
                n_payouts = preview.get("payouts_found", 0)
                n_with_game = preview.get("items_with_game", 0)
                n_no_game = preview.get("items_no_game", 0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("DK Wagers Found", n_wagers)
                c2.metric("Win Payouts Found", n_payouts)
                c3.metric("Games Auto-Matched", n_with_game)
                c4.metric("No Game Found", n_no_game, delta_color="inverse")

                items = preview.get("items", [])

                if not items:
                    st.warning("No wagers parsed from CSV.")
                else:
                    if n_no_game > 0:
                        st.warning(
                            f"{n_no_game} wager(s) had no matching game in the database. "
                            "Check the Games Reference table below for the correct Game ID, "
                            "enter it in the 'Game ID' column, then confirm."
                        )

                    st.write(
                        "**Review matched games below.** "
                        "Edit the **Game ID** column to correct any wrong matches. "
                        "Uncheck rows you don't want to import."
                    )

                    df_rows = []
                    for it in items:
                        ts = it["dk_timestamp"][:16].replace("T", " ")
                        game_label = it.get("suggested_game_label") or "— no game found —"
                        df_rows.append({
                            "✓": it.get("suggested_game_id") is not None,
                            "DK Time (UTC)": ts,
                            "Wager $": f"${it['dk_amount']:.2f}",
                            "Result": "WIN" if it["outcome"] == 1 else ("LOSS" if it["outcome"] == 0 else "PENDING"),
                            "P&L": f"${it['profit_dollars']:+.2f}" if it["outcome"] is not None else "—",
                            "Game ID": int(it["suggested_game_id"]) if it.get("suggested_game_id") else 0,
                            "Matched Game": game_label,
                        })

                    df_preview = pd.DataFrame(df_rows)
                    edited = st.data_editor(
                        df_preview,
                        column_config={
                            "✓": st.column_config.CheckboxColumn("Import?", default=True),
                            "Game ID": st.column_config.NumberColumn(
                                "Game ID",
                                help="Edit to correct the game. Use the Games Reference table below to look up IDs.",
                                min_value=0,
                                step=1,
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                        disabled=["DK Time (UTC)", "Wager $", "Result", "P&L", "Matched Game"],
                    )

                    st.session_state["dk_direct_items"] = items
                    st.session_state["dk_direct_game_ids"] = edited["Game ID"].tolist()
                    st.session_state["dk_direct_checked"] = edited["✓"].tolist()

                    wins = sum(1 for it in items if it["outcome"] == 1)
                    losses = sum(1 for it in items if it["outcome"] == 0)
                    pending = sum(1 for it in items if it["outcome"] is None)
                    total_pl = sum(it["profit_dollars"] for it in items if it["outcome"] is not None)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Wins", wins)
                    s2.metric("Losses", losses)
                    s3.metric("Pending", pending)
                    s4.metric("Total P&L", f"${total_pl:+.2f}")

                    # Games reference table
                    with st.expander("Games Reference — look up Game IDs to correct matches"):
                        all_dates = sorted({it["dk_timestamp"][:10] for it in items})
                        ref_data = api_get("/api/games/recent", {"days_back": 60, "days_ahead": 2})
                        if ref_data and ref_data.get("games"):
                            ref_df = pd.DataFrame([
                                {
                                    "ID": g["id"],
                                    "Date": g["game_date"][:10],
                                    "Matchup": g["matchup"],
                                    "Completed": g.get("completed", False),
                                }
                                for g in ref_data["games"]
                            ])
                            st.dataframe(ref_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No games found in database.")

    # Confirm button (outside the file-uploader block so it persists)
    if "dk_direct_items" in st.session_state:
        all_items = st.session_state["dk_direct_items"]
        game_ids  = st.session_state.get("dk_direct_game_ids", [it.get("suggested_game_id", 0) for it in all_items])
        checked   = st.session_state.get("dk_direct_checked", [True] * len(all_items))

        payload_items = []
        for it, gid, keep in zip(all_items, game_ids, checked):
            try:
                gid_int = int(gid) if gid else 0
            except (TypeError, ValueError):
                gid_int = 0
            if keep and gid_int > 0:
                payload_items.append({
                    "dk_wager_id": it["dk_wager_id"],
                    "dk_amount": it["dk_amount"],
                    "dk_timestamp": it["dk_timestamp"],
                    "game_id": gid_int,
                    "outcome": it["outcome"],
                    "profit_dollars": it["profit_dollars"],
                })

        if payload_items:
            if st.button(
                f"Confirm & Import {len(payload_items)} Bet(s)",
                type="primary",
                key="dk_direct_confirm_btn",
            ):
                with st.spinner("Creating real bet records..."):
                    result = api_post("/admin/dk/direct-confirm", {"items": payload_items}, timeout=60)

                if result and result.get("status") == "ok":
                    applied  = result["applied"]
                    pending_ = result.get("pending_added", 0)
                    st.success(
                        f"Imported {applied} settled + {pending_} pending bet(s) — "
                        f"{result['wins']} wins, {result['losses']} losses, "
                        f"P&L: ${result['total_profit_dollars']:+.2f}"
                    )
                    if result.get("errors"):
                        st.warning(f"Some errors: {result['errors']}")
                    for key in ("dk_direct_items", "dk_direct_game_ids", "dk_direct_checked"):
                        st.session_state.pop(key, None)
                    st.rerun()
                else:
                    errors = result.get("errors", []) if result else []
                    st.error(f"Import failed. Errors: {errors}")
        elif "dk_direct_items" in st.session_state:
            st.info("No rows selected with a valid Game ID — nothing to import.")

# --------------------------------------------------------------------------
# TAB 2: PAPER TRADE MATCHING (original flow)
# --------------------------------------------------------------------------
with dk_tab2:
    st.write(
        "Match DK wagers to existing model paper trades. "
        "Useful if the nightly analysis ran and you want to link real results "
        "to the model's recommendations."
    )

    paper_file = st.file_uploader(
        "Upload DraftKings transaction history (.csv)",
        type=["csv"],
        key="dk_paper_uploader",
    )

    if paper_file is not None:
        csv_content = paper_file.read().decode("utf-8", errors="replace")
        st.success(f"File loaded: {paper_file.name} ({len(csv_content):,} bytes)")

        if st.button("Preview Matches", type="primary", key="dk_paper_preview_btn"):
            with st.spinner("Parsing CSV and matching to paper trades..."):
                preview = api_post("/admin/dk/preview", {"csv_content": csv_content}, timeout=30)

            if preview is None:
                st.error("Preview failed — check API logs.")
            else:
                n_wagers = preview.get("wagers_found", 0)
                n_payouts = preview.get("payouts_found", 0)
                n_matches = len(preview.get("matches", []))
                n_unmatched = preview.get("unmatched_wagers", 0)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("DK Wagers Found", n_wagers)
                m2.metric("Win Payouts Found", n_payouts)
                m3.metric("Matched to Paper Trades", n_matches)
                m4.metric("Unmatched Wagers", n_unmatched)

                matches = preview.get("matches", [])

                if not matches:
                    st.warning(
                        "No matches found. "
                        "Use the Direct Import tab instead, which does not require paper trades."
                    )
                else:
                    st.write("**Proposed Matches** — review and uncheck any incorrect matches:")

                    df = pd.DataFrame([
                        {
                            "✓": True,
                            "Pick": m["pick"],
                            "BetLog Time (UTC)": m["bet_log_timestamp"][:16].replace("T", " "),
                            "Model $": f"${m['bet_log_dollars']:.2f}",
                            "DK Wager $": f"${m['dk_wager_amount']:.2f}",
                            "DK Time (UTC)": m["dk_wager_timestamp"][:16].replace("T", " "),
                            "Outcome": "WIN" if m["outcome"] == 1 else ("LOSS" if m["outcome"] == 0 else "PENDING"),
                            "P&L": f"${m['profit_dollars']:+.2f}" if m["outcome"] is not None else "—",
                            "Confidence": m["confidence"],
                        }
                        for m in matches
                    ])

                    edited = st.data_editor(
                        df,
                        column_config={"✓": st.column_config.CheckboxColumn("Apply?", default=True)},
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.session_state["dk_preview_matches"] = matches
                    st.session_state["dk_preview_checked"] = edited["✓"].tolist()

                    wins = sum(1 for m in matches if m["outcome"] == 1)
                    losses = sum(1 for m in matches if m["outcome"] == 0)
                    pending = sum(1 for m in matches if m["outcome"] is None)
                    total_pl = sum(m["profit_dollars"] for m in matches if m["outcome"] is not None)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Wins", wins)
                    s2.metric("Losses", losses)
                    s3.metric("Pending", pending)
                    s4.metric("Total P&L", f"${total_pl:+.2f}")

    # Confirm button (always visible if preview has run)
    if "dk_preview_matches" in st.session_state:
        matches_to_apply = st.session_state["dk_preview_matches"]
        checked = st.session_state.get("dk_preview_checked", [True] * len(matches_to_apply))
        selected = [m for m, keep in zip(matches_to_apply, checked) if keep]

        if selected:
            if st.button(f"Confirm & Apply {len(selected)} Match(es)", type="primary", key="dk_paper_confirm_btn"):
                with st.spinner("Applying matches to database..."):
                    result = api_post("/admin/dk/confirm", {"matches": selected}, timeout=30)
                if result and result.get("status") == "ok":
                    st.success(
                        f"Applied {result['applied']} matches — "
                        f"{result['wins']} wins, {result['losses']} losses, "
                        f"P&L: ${result['total_profit_dollars']:+.2f}"
                    )
                    del st.session_state["dk_preview_matches"]
                    del st.session_state["dk_preview_checked"]
                    st.rerun()
                else:
                    errors = result.get("errors", []) if result else []
                    st.error(f"Import failed. Errors: {errors}")

st.markdown("---")

# --- Scheduler jobs ---
st.subheader("Scheduled Jobs")
sched = api_get("/admin/scheduler/status")

if sched and sched.get("jobs"):
    for job in sched["jobs"]:
        next_run = job.get("next_run", "—")
        if next_run and next_run != "—":
            next_run = next_run[:19].replace("T", " ")
        status_icon = "running" if job.get("pending", False) else "idle"
        st.write(f"- **{job['name']}** | Next: `{next_run}` | Status: {status_icon}")
else:
    st.warning("Could not fetch scheduler status. Is the admin key configured?")

st.markdown("---")

# --- Portfolio exposure / drawdown ---
st.subheader("Portfolio Status")
portfolio = api_get("/admin/portfolio/status")

if portfolio:
    g1, g2 = st.columns(2)

    with g1:
        exposure = portfolio.get("current_exposure_pct", 0)
        max_exposure = get_float_env("MAX_DAILY_EXPOSURE_PCT", "15.0")
        st.metric("Current Exposure", f"{exposure:.1f}%")
        st.progress(min(exposure / max_exposure, 1.0) if max_exposure > 0 else 0)
        st.caption(f"Cap: {max_exposure:.0f}%")

    with g2:
        drawdown = portfolio.get("current_drawdown_pct", 0)
        max_dd = get_float_env("MAX_DRAWDOWN_PCT", "15.0")
        st.metric("Current Drawdown", f"{drawdown:.1f}%")
        st.progress(min(drawdown / max_dd, 1.0) if max_dd > 0 else 0)
        st.caption(f"Circuit breaker: {max_dd:.0f}%")

    if portfolio.get("circuit_breaker_active"):
        st.error("Circuit breaker ACTIVE — all new bets paused.")

    st.metric("Open Positions", portfolio.get("open_positions", 0))
else:
    st.warning("Portfolio data unavailable.")

st.markdown("---")

# --- Environment config display ---
st.subheader("Environment Configuration")

config_vars = {
    "STARTING_BANKROLL": os.getenv("STARTING_BANKROLL", "1000"),
    "MAX_DAILY_EXPOSURE_PCT": os.getenv("MAX_DAILY_EXPOSURE_PCT", "15.0"),
    "MAX_DRAWDOWN_PCT": os.getenv("MAX_DRAWDOWN_PCT", "15.0"),
    "MAX_KELLY_FRACTION": os.getenv("MAX_KELLY_FRACTION", "0.20"),
    "ODDS_API_REGIONS": os.getenv("ODDS_API_REGIONS", "us,eu"),
    "MIN_BET_EDGE": os.getenv("MIN_BET_EDGE", "2.5"),
    "BASE_SD": os.getenv("BASE_SD", "11.0"),
    "HOME_ADVANTAGE": os.getenv("HOME_ADVANTAGE", "3.09"),
    "ENVIRONMENT": os.getenv("ENVIRONMENT", "production"),
}

for var, val in config_vars.items():
    st.write(f"- `{var}` = **{val}**")
