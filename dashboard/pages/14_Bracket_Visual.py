"""
Bracket Visual — full 68-team bracket with model-projected winners.

Improvements over v1:
- Standalone: runs its own Monte Carlo simulation if page 13 data is absent
- Probability bars: champion %, F4 %, S16 % shown per team
- Better HTML bracket: seed-colored pills, win% on winner, upset highlights
- Key Insights: top Cinderellas, most dangerous upsets, champion odds by tier
- Chaos slider now drives actual stochastic simulation (re-seeds RNG)

Layout (classic NCAA bracket):
    East   (top-left)  advances right ──────────┐
                                                  ├─ F4 Left  ─┐
    South  (bot-right) advances left ──────────┘               │
                                                                ├── CHAMPION
    West   (top-right) advances left ──────────┐               │
                                                  ├─ F4 Right ─┘
    Midwest (bot-left) advances right ──────────┘

F4 pairings: South vs East · West vs Midwest
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json, math, random
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from typing import Dict, List, Optional, Tuple

st.set_page_config(
    page_title="Bracket Visual | CBB Edge",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from backend.tournament.matchup_predictor import TournamentTeam, predict_game
    from backend.tournament.bracket_simulator import R64_SEED_ORDER, FF_PAIRINGS
    TOURNAMENT_OK = True
except ImportError as e:
    TOURNAMENT_OK = False
    st.error(f"Tournament module not available: {e}")
    st.stop()

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bracket_2026.json"
REGIONS = ["east", "south", "west", "midwest"]


# ---------------------------------------------------------------------------
# Bracket loading
# ---------------------------------------------------------------------------
def load_bracket_from_disk() -> dict:
    if not BRACKET_PATH.exists():
        raise FileNotFoundError(f"bracket_2026.json not found at {BRACKET_PATH}")
    with open(BRACKET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    bracket = {}
    for region in REGIONS:
        if region not in raw:
            continue
        bracket[region] = [
            TournamentTeam(
                name=t["name"], seed=t["seed"], region=region,
                composite_rating=t.get("composite_rating"),
                kp_adj_em=t.get("kp_adj_em"),
                bt_adj_em=t.get("bt_adj_em"),
                pace=t.get("pace", 68.0),
                three_pt_rate=t.get("three_pt_rate", 0.35),
                def_efg_pct=t.get("def_efg_pct", 0.50),
                conference=t.get("conference", ""),
                tournament_exp=t.get("tournament_exp", 0.70),
            )
            for t in raw[region]
        ]
    return bracket


def load_bracket_from_session() -> Optional[dict]:
    if "bracket_teams" not in st.session_state:
        return None
    bracket = {}
    for region in REGIONS:
        entries = st.session_state.bracket_teams.get(region, [])
        if not entries or not any(e.get("name", "").strip() for e in entries):
            return None
        teams = []
        for e in entries:
            name = e.get("name", "").strip() or f"{region.title()}-Seed{e['seed']}"
            teams.append(TournamentTeam(
                name=name, seed=e["seed"], region=region,
                composite_rating=e.get("composite_rating"),
                kp_adj_em=e.get("kp_adj_em"),
                bt_adj_em=e.get("bt_adj_em"),
                pace=e.get("pace", 68.0),
                three_pt_rate=e.get("three_pt_rate", 0.35),
                def_efg_pct=e.get("def_efg_pct", 0.50),
                conference=e.get("conference", ""),
                tournament_exp=e.get("tournament_exp", 0.70),
            ))
        bracket[region] = teams
    return bracket


# ---------------------------------------------------------------------------
# Monte Carlo simulation (lightweight, runs in-page)
# ---------------------------------------------------------------------------
def _sim_region(teams: List[TournamentTeam], rng: random.Random, round_offset: int = 0):
    """Simulate one region through R64→E8. Returns (winner, r32_names, s16_names, e8_name)."""
    seed_map = {t.seed: t for t in teams}
    slots = [seed_map[s] for s in R64_SEED_ORDER if s in seed_map]
    r32, s16 = [], []
    for rnd in [1, 2, 3, 4]:
        nxt = []
        for i in range(0, len(slots), 2):
            ta, tb = slots[i], slots[i + 1]
            p, _, _ = predict_game(ta, tb, rnd)
            nxt.append(ta if rng.random() < p else tb)
        slots = nxt
        if rnd == 2: r32 = [t.name for t in slots]
        elif rnd == 3: s16 = [t.name for t in slots]
    return slots[0], r32, s16, slots[0].name


def run_quick_mc(bracket: dict, n_sims: int = 5000, chaos_seed: int = 42) -> Dict:
    """
    Run n_sims Monte Carlo bracket simulations.
    Returns {team_name: {champ, f4, e8, s16, r32}} probability dicts.
    """
    all_teams = [t for ts in bracket.values() for t in ts]
    counts = {t.name: {"champ": 0, "f4": 0, "e8": 0, "s16": 0, "r32": 0}
              for t in all_teams}
    rng = random.Random(chaos_seed)

    for _ in range(n_sims):
        region_winners: Dict[str, TournamentTeam] = {}
        for region, teams in bracket.items():
            winner, r32, s16, _ = _sim_region(teams, rng)
            region_winners[region] = winner
            for n in r32:
                if n in counts: counts[n]["r32"] += 1
            for n in s16:
                if n in counts: counts[n]["s16"] += 1
            if winner.name in counts:
                counts[winner.name]["e8"] += 1

        # Final Four
        finalists = []
        for reg_a, reg_b in FF_PAIRINGS:
            ta = region_winners.get(reg_a)
            tb = region_winners.get(reg_b)
            if ta is None or tb is None:
                continue
            p, _, _ = predict_game(ta, tb, 5)
            winner = ta if rng.random() < p else tb
            if ta.name in counts: counts[ta.name]["f4"] += 1
            if tb.name in counts: counts[tb.name]["f4"] += 1
            finalists.append(winner)

        if len(finalists) >= 2:
            p, _, _ = predict_game(finalists[0], finalists[1], 6)
            champ = finalists[0] if rng.random() < p else finalists[1]
            if champ.name in counts:
                counts[champ.name]["champ"] += 1

    return {name: {k: v / n_sims for k, v in c.items()} for name, c in counts.items()}


# ---------------------------------------------------------------------------
# Deterministic bracket generator (for visual)
# ---------------------------------------------------------------------------
def generate_bracket(bracket: dict, chaos_level: float = 0.0,
                     rng_seed: int = 42) -> Tuple[dict, list, dict, list]:
    """
    Generate a predicted bracket with optional probabilistic upset sampling.

    Chaos calibration is anchored to historical NCAA tournament upset rates:
      chaos=0.0 → chalk (model favourite always wins)
      chaos=0.5 → pure model-probability sampling, which reproduces the
                  historical upset rates (~31% of games per round)
      chaos=1.0 → upset probabilities amplified up to 2× (capped at 50%),
                  producing a chaotic but still seeding-aware bracket

    The old threshold formula (0.5 + (0.5-p)*(1-chaos)) was NOT calibrated:
    at chaos=0.5 it gave a 1v16 game a 73% upset chance instead of ~3%.
    The corrected approach:
      effective_upset_p = upset_p * (chaos / 0.5)        for chaos ≤ 0.5
      effective_upset_p = min(0.5, upset_p*(1+(chaos-0.5)/0.5)) for chaos > 0.5

    Returns (region_rounds, ff, champ, upsets)
    where region_rounds[region][round_num] = list of matchup dicts
    """
    rng = random.Random(rng_seed)
    region_rounds = {}

    def pick_winner(ta, tb, rnd, chaos):
        p, _, _ = predict_game(ta, tb, rnd)
        if chaos == 0.0:
            return (ta, tb, p) if p >= 0.5 else (tb, ta, 1.0 - p)

        # Always identify favourite by model probability
        if p >= 0.5:
            fav, dog, fav_p = ta, tb, p
        else:
            fav, dog, fav_p = tb, ta, 1.0 - p

        upset_p = 1.0 - fav_p  # model's raw upset probability

        # Scale effective upset probability based on chaos level:
        #   chaos=0.5 → effective = upset_p  (historically calibrated)
        #   chaos<0.5 → blend toward chalk
        #   chaos>0.5 → amplify toward 50/50 (but never exceed it)
        if chaos <= 0.5:
            effective_upset_p = upset_p * (chaos / 0.5)
        else:
            t = (chaos - 0.5) / 0.5
            effective_upset_p = min(0.5, upset_p * (1.0 + t))

        if rng.random() < effective_upset_p:
            return dog, fav, upset_p
        return fav, dog, fav_p

    upsets_global = []

    for region, teams in bracket.items():
        seed_map = {t.seed: t for t in teams}
        slots = [seed_map[s] for s in R64_SEED_ORDER if s in seed_map]

        rounds = {0: [(slots[i], slots[i + 1]) for i in range(0, len(slots), 2)]}
        current = slots

        for rnd in [1, 2, 3, 4]:
            matchups, nxt = [], []
            for i in range(0, len(current), 2):
                ta, tb = current[i], current[i + 1]
                winner, loser, prob = pick_winner(ta, tb, rnd, chaos_level)
                is_upset = winner.seed > loser.seed
                if is_upset:
                    upsets_global.append({
                        "region": region, "round": rnd,
                        "winner": winner.name, "winner_seed": winner.seed,
                        "loser": loser.name, "loser_seed": loser.seed,
                        "prob": prob,
                    })
                matchups.append({"ta": ta, "tb": tb, "winner": winner,
                                 "loser": loser, "prob": prob, "is_upset": is_upset})
                nxt.append(winner)
            rounds[rnd] = matchups
            current = nxt

        region_rounds[region] = rounds

    # Final Four
    ff = []
    for reg_a, reg_b in FF_PAIRINGS:
        ta = region_rounds[reg_a][4][0]["winner"]
        tb = region_rounds[reg_b][4][0]["winner"]
        winner, loser, prob = pick_winner(ta, tb, 5, chaos_level)
        ff.append({"ta": ta, "tb": tb, "winner": winner, "loser": loser,
                   "prob": prob, "is_upset": winner.seed > loser.seed,
                   "regions": (reg_a, reg_b)})

    ta, tb = ff[0]["winner"], ff[1]["winner"]
    winner, loser, prob = pick_winner(ta, tb, 6, chaos_level)
    champ = {"ta": ta, "tb": tb, "winner": winner, "loser": loser,
             "prob": prob, "is_upset": winner.seed > loser.seed}

    return region_rounds, ff, champ, upsets_global


# ---------------------------------------------------------------------------
# HTML bracket renderer (improved)
# ---------------------------------------------------------------------------
TEAM_W = 155
TEAM_H = 26
ROUND_GAP = 5

SEED_COLORS = {
    (1, 2):   ("#1565C0", "#E3F2FD"),
    (3, 4):   ("#2E7D32", "#E8F5E9"),
    (5, 8):   ("#BF360C", "#FBE9E7"),
    (9, 12):  ("#6A1B9A", "#F3E5F5"),
    (13, 16): ("#4E342E", "#EFEBE9"),
}

def _seed_style(seed: int) -> Tuple[str, str]:
    for (lo, hi), (fg, bg) in SEED_COLORS.items():
        if lo <= seed <= hi:
            return fg, bg
    return "#424242", "#F5F5F5"


def _team_box(team: TournamentTeam, is_winner: bool,
              win_prob: Optional[float] = None,
              is_champ: bool = False,
              mc_champ_pct: Optional[float] = None) -> str:
    fg, bg_seed = _seed_style(team.seed)
    if is_champ:
        bg = "#FFF9C4"; border = "2px solid #F9A825"
    elif is_winner:
        bg = "#E8F5E9"; border = "1px solid #66BB6A"
    else:
        bg = "#F5F5F5"; border = "1px solid #CFD8DC"
    opacity = "1.0" if is_winner else "0.45"
    fw = "600" if is_winner else "400"

    # Win probability badge on winner
    prob_html = ""
    if win_prob and is_winner:
        prob_html = (f"<span style='font-size:9px;color:#1565C0;font-weight:bold;"
                     f"margin-left:4px;white-space:nowrap'>{win_prob:.0%}</span>")

    # MC championship % as right-side micro-bar (top seeds only)
    mc_html = ""
    if mc_champ_pct and mc_champ_pct >= 0.005 and is_winner:
        bar_w = max(2, int(mc_champ_pct * 60))
        mc_html = (f"<span style='display:inline-block;width:{bar_w}px;height:6px;"
                   f"background:#1565C0;border-radius:2px;margin-left:4px;opacity:0.7;"
                   f"vertical-align:middle' title='{mc_champ_pct:.1%} MC champ'></span>")

    seed_pill = (f"<span style='background:{fg};color:white;border-radius:3px;"
                 f"padding:0 4px;font-size:9px;font-weight:bold;min-width:18px;"
                 f"display:inline-block;text-align:center;margin-right:3px'>#{team.seed}</span>")

    name_span = (f"<span style='font-weight:{fw};overflow:hidden;text-overflow:ellipsis;"
                 f"white-space:nowrap;flex:1;font-size:11px'>{team.name}</span>")

    return (
        f'<div style="width:{TEAM_W}px;height:{TEAM_H}px;border:{border};background:{bg};'
        f'display:flex;align-items:center;padding:0 4px;box-sizing:border-box;'
        f'opacity:{opacity};font-family:\'Segoe UI\',Arial,sans-serif;border-radius:2px">'
        f'{seed_pill}{name_span}{prob_html}{mc_html}'
        f'</div>'
    )


def _matchup_block(matchup: dict, margin_px: int,
                   mc_probs: Optional[Dict] = None) -> str:
    ta, tb = matchup["ta"], matchup["tb"]
    winner = matchup["winner"]
    prob = matchup["prob"]
    a_wins = winner is ta
    ta_mc = (mc_probs or {}).get(ta.name, {}).get("champ")
    tb_mc = (mc_probs or {}).get(tb.name, {}).get("champ")
    ta_html = _team_box(ta, a_wins, prob if a_wins else None, mc_champ_pct=ta_mc if a_wins else None)
    tb_html = _team_box(tb, not a_wins, prob if not a_wins else None, mc_champ_pct=tb_mc if not a_wins else None)
    upset_flag = ""
    if matchup.get("is_upset"):
        upset_flag = "<span style='font-size:8px;color:#D32F2F;margin-left:2px'>⚡</span>"
    return (
        f'<div style="margin:{margin_px}px 0 0 0">'
        f'{ta_html}'
        f'<div style="position:relative">{tb_html}{upset_flag}</div>'
        f'</div>'
    )


def _round_label(text: str) -> str:
    return (f'<div style="text-align:center;font-size:9px;font-weight:bold;'
            f'color:#546E7A;margin-bottom:3px;letter-spacing:0.5px">{text}</div>')


def _round_col(matchups: list, round_num: int, label: str = "",
               mc_probs: Optional[Dict] = None) -> str:
    margin = (2 ** (round_num - 1) - 1) * TEAM_H
    lbl = _round_label(label) if label else ""
    blocks = "".join(_matchup_block(m, margin, mc_probs) for m in matchups)
    return (f'<div style="display:flex;flex-direction:column;margin:0 {ROUND_GAP}px">'
            f'{lbl}{blocks}</div>')


def _r64_col(region: str, region_rounds: dict, mc_probs: Optional[Dict] = None) -> str:
    pairs = region_rounds[region][0]
    matchups_r1 = region_rounds[region][1]
    blocks = []
    for (ta, tb), m in zip(pairs, matchups_r1):
        a_wins = m["winner"] is ta
        ta_mc = (mc_probs or {}).get(ta.name, {}).get("champ")
        tb_mc = (mc_probs or {}).get(tb.name, {}).get("champ")
        ta_html = _team_box(ta, a_wins, m["prob"] if a_wins else None,
                            mc_champ_pct=ta_mc if a_wins else None)
        tb_html = _team_box(tb, not a_wins, m["prob"] if not a_wins else None,
                            mc_champ_pct=tb_mc if not a_wins else None)
        upset_flag = "<span style='font-size:8px;color:#D32F2F'>⚡</span>" if m.get("is_upset") else ""
        blocks.append(f'<div style="margin:0">{ta_html}<div style="position:relative">'
                      f'{tb_html}{upset_flag}</div></div>')
    return (f'<div style="display:flex;flex-direction:column;margin:0 {ROUND_GAP}px">'
            f'{_round_label("R64")}{"".join(blocks)}</div>')


def _region_ltr(region: str, region_rounds: dict, label: str,
                mc_probs: Optional[Dict] = None) -> str:
    lbl = (f'<div style="text-align:center;font-size:11px;font-weight:bold;color:#1565C0;'
           f'margin-bottom:5px;letter-spacing:0.8px">{label}</div>')
    r64 = _r64_col(region, region_rounds, mc_probs)
    r32 = _round_col(region_rounds[region][2], 2, "R32", mc_probs)
    s16 = _round_col(region_rounds[region][3], 3, "S16", mc_probs)
    e8  = _round_col(region_rounds[region][4], 4, "E8",  mc_probs)
    return (f'<div>{lbl}<div style="display:flex;flex-direction:row;align-items:flex-start">'
            f'{r64}{r32}{s16}{e8}</div></div>')


def _region_rtl(region: str, region_rounds: dict, label: str,
                mc_probs: Optional[Dict] = None) -> str:
    lbl = (f'<div style="text-align:center;font-size:11px;font-weight:bold;color:#1565C0;'
           f'margin-bottom:5px;letter-spacing:0.8px">{label}</div>')
    r64 = _r64_col(region, region_rounds, mc_probs)
    r32 = _round_col(region_rounds[region][2], 2, "R32", mc_probs)
    s16 = _round_col(region_rounds[region][3], 3, "S16", mc_probs)
    e8  = _round_col(region_rounds[region][4], 4, "E8",  mc_probs)
    return (f'<div>{lbl}<div style="display:flex;flex-direction:row;align-items:flex-start">'
            f'{e8}{s16}{r32}{r64}</div></div>')


def _ff_center(ff: list, champ: dict) -> str:
    def ff_slot(m: dict, label: str) -> str:
        ta, tb = m["ta"], m["tb"]
        a_wins = m["winner"] is ta
        ta_html = _team_box(ta, a_wins, m["prob"] if a_wins else None)
        tb_html = _team_box(tb, not a_wins, m["prob"] if not a_wins else None)
        lbl = (f'<div style="text-align:center;font-size:9px;font-weight:bold;'
               f'color:#E65100;margin-bottom:3px">{label}</div>')
        return f'<div style="margin-bottom:10px">{lbl}{ta_html}{tb_html}</div>'

    champ_html = _team_box(champ["winner"], True, champ["prob"], is_champ=True)
    ru_html = _team_box(champ["loser"], False)
    champ_lbl = (f'<div style="text-align:center;font-size:10px;font-weight:bold;'
                 f'color:#B71C1C;margin-bottom:3px">🏆 CHAMPION</div>')
    ru_lbl = (f'<div style="text-align:center;font-size:9px;color:#78909C;'
              f'margin-bottom:2px">Runner-Up</div>')

    ff0 = ff_slot(ff[0], f"F4 · {ff[0]['regions'][0].title()} vs {ff[0]['regions'][1].title()}")
    ff1 = ff_slot(ff[1], f"F4 · {ff[1]['regions'][0].title()} vs {ff[1]['regions'][1].title()}")

    inner = (f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center">'
             f'{ff0}<div style="height:12px"></div>'
             f'{champ_lbl}{champ_html}'
             f'<div style="height:4px"></div>{ru_lbl}{ru_html}'
             f'<div style="height:12px"></div>{ff1}'
             f'</div>')
    return (f'<div style="width:{TEAM_W + 20}px;display:flex;flex-direction:column;'
            f'justify-content:center;padding:0 {ROUND_GAP}px;'
            f'border-left:2px dashed #B0BEC5;border-right:2px dashed #B0BEC5">'
            f'{inner}</div>')


def render_bracket_html(region_rounds: dict, ff: list, champ: dict,
                        mc_probs: Optional[Dict] = None,
                        chaos_level: float = 0.0) -> str:
    east    = _region_ltr("east",    region_rounds, "EAST",    mc_probs)
    midwest = _region_ltr("midwest", region_rounds, "MIDWEST", mc_probs)
    west    = _region_rtl("west",    region_rounds, "WEST",    mc_probs)
    south   = _region_rtl("south",   region_rounds, "SOUTH",   mc_probs)
    center  = _ff_center(ff, champ)

    row1 = (f'<div style="display:flex;flex-direction:row;align-items:flex-start;margin-bottom:14px">'
            f'{east}{center}{west}</div>')
    spacer_w = TEAM_W + 20 + 2 * ROUND_GAP
    row2 = (f'<div style="display:flex;flex-direction:row;align-items:flex-start">'
            f'{midwest}<div style="width:{spacer_w}px"></div>{south}</div>')

    if chaos_level == 0.0:
        mode_txt, mode_color = "CHALK BRACKET — Model Favorites", "#1565C0"
    elif chaos_level <= 0.4:
        mode_txt, mode_color = "MODEL BRACKET — Probability-Weighted", "#2E7D32"
    elif chaos_level <= 0.7:
        mode_txt, mode_color = "STYLE-AWARE BRACKET — Upsets Possible", "#E65100"
    else:
        mode_txt, mode_color = "CHAOS BRACKET — March Madness!", "#B71C1C"

    mc_legend = ""
    if mc_probs:
        mc_legend = ("<span style='margin-left:16px'>"
                     "<span style='display:inline-block;width:20px;height:6px;background:#1565C0;"
                     "border-radius:2px;vertical-align:middle;margin-right:4px'></span>"
                     "Blue bar = MC champion %</span>")

    legend_items = "".join(
        f'<span style="margin-right:10px">'
        f'<span style="background:{fg};color:white;padding:1px 5px;border-radius:3px;font-size:9px">'
        f'#{lo}–{hi}</span></span>'
        for (lo, hi), (fg, _) in SEED_COLORS.items()
    )
    legend = (f'<div style="display:flex;align-items:center;margin-bottom:10px;'
              f'font-size:10px;font-family:Arial;flex-wrap:wrap;gap:4px">'
              f'{legend_items}'
              f'<span style="margin-left:8px">⚡ = upset</span>'
              f'<span style="margin-left:8px">% on winner = matchup win prob</span>'
              f'{mc_legend}</div>')

    header = (f'<div style="text-align:center;font-size:15px;font-weight:bold;'
              f'color:{mode_color};margin-bottom:8px;font-family:Arial;letter-spacing:0.5px">'
              f'2026 NCAA TOURNAMENT — {mode_txt}'
              f'<span style="font-size:10px;font-weight:normal;color:#78909C;margin-left:10px">'
              f'V9.1 model · win% on winner · ⚡ = upset</span></div>')

    return (f'<html><head><meta charset="utf-8"></head>'
            f'<body style="margin:0;padding:10px;background:#FAFAFA">'
            f'{header}{legend}'
            f'<div style="overflow-x:auto">{row1}{row2}</div>'
            f'</body></html>')


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🏆 2026 NCAA Tournament Bracket")
st.caption("V9.1 model · 5-year calibrated upset rates · Monte Carlo probability bars")

# Load bracket
bracket = load_bracket_from_session()
source = "page 13 session state"
if bracket is None:
    try:
        bracket = load_bracket_from_disk()
        source = "data/bracket_2026.json"
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Go to page 13 (Tournament Bracket) and load/enter teams first.")
        st.stop()

named = sum(1 for ts in bracket.values() for t in ts
            if not t.name.startswith(f"{t.region.title()}-Seed"))
st.caption(f"Source: {source} | {named}/64 teams named")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Bracket Controls")
    chaos_level = st.slider(
        "Chaos Level",
        min_value=0.0, max_value=1.0,
        value=st.session_state.get("chaos_level", 0.0),
        step=0.1,
        help="0 = chalk (favorites always win) · 1 = full stochastic sampling"
    )
    st.session_state.chaos_level = chaos_level

    n_sims = st.selectbox("MC Simulations", [1000, 3000, 5000, 10000],
                          index=2, help="More sims = smoother probability bars")
    show_mc = st.toggle("Show MC probability bars", value=True)
    pool_optimal = st.toggle("Pool Optimal Mode", value=False,
                             help="Surgically picks 2×12v5 + 1×11v6 upsets; all 1-seeds to Final Four")
    rng_seed = st.number_input("Random seed", value=42, min_value=0,
                               help="Change for a different stochastic bracket")

    if st.button("🎲 Regenerate Bracket", type="primary", use_container_width=True):
        for key in ("bracket_cache", "mc_cache"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.subheader("Live Data (BallDontLie)")

    bdl_date = st.date_input("Game date", value=None,
                              help="Fetch odds for this date (default: today)")
    col_odds, col_stats = st.columns(2)

    with col_odds:
        if st.button("Refresh Odds", use_container_width=True,
                     help="Pull live moneylines from BDL and patch market_ml"):
            import json as _json
            from datetime import date as _date
            _target = bdl_date.isoformat() if bdl_date else _date.today().isoformat()
            try:
                from backend.services.balldontlie import BallDontLieClient
                _client = BallDontLieClient()
                with open(BRACKET_PATH, encoding="utf-8") as _f:
                    _raw = _json.load(_f)

                _odds = _client.get_odds_by_date(_target)
                _games = _client.get_live_tournament_games(_target)
                _patched = 0
                _game_odds: dict = {}
                for _rec in _odds:
                    _gid = _rec.get("game_id") or (_rec.get("game") or {}).get("id")
                    if _gid:
                        _game_odds.setdefault(_gid, []).append(_rec)

                for _g in _games:
                    _home = (_g.get("home_team") or {})
                    _away = (_g.get("visitor_team") or _g.get("away_team") or {})
                    _hn = _home.get("name", "")
                    _an = _away.get("name", "")
                    _gid = _g.get("id")
                    _recs = _game_odds.get(_gid, [])
                    if not _recs:
                        continue
                    _ml = _client.extract_market_ml(_recs, _hn, _an)
                    for _nm, _ml_val in [(_hn, _ml.get("home_ml")), (_an, _ml.get("away_ml"))]:
                        if not _nm or _ml_val is None:
                            continue
                        for _region in ["east", "west", "south", "midwest"]:
                            for _t in _raw.get(_region, []):
                                if _t.get("name", "").lower() == _nm.lower():
                                    _t["market_ml"] = int(_ml_val)
                                    _patched += 1

                if _patched > 0:
                    with open(BRACKET_PATH, "w", encoding="utf-8") as _f:
                        _json.dump(_raw, _f, indent=2, ensure_ascii=False)
                    for _key in ("bracket_cache", "mc_cache"):
                        st.session_state.pop(_key, None)
                    st.success(f"Patched {_patched} ML odds. Reloading...")
                    st.rerun()
                else:
                    st.info(f"No odds found for {_target}")
            except ValueError as _e:
                st.error(f"API key missing: {_e}")
            except Exception as _e:
                st.error(f"BDL error: {_e}")

    with col_stats:
        if st.button("Refresh Stats", use_container_width=True,
                     help="Pull season pace/3PT stats from BDL"):
            import json as _json
            try:
                from backend.services.balldontlie import BallDontLieClient
                _client = BallDontLieClient()
                with open(BRACKET_PATH, encoding="utf-8") as _f:
                    _raw = _json.load(_f)

                _stats = _client.get_team_season_stats()
                _by_name = {}
                for _row in _stats:
                    _t = _row.get("team", {})
                    _n = (_t.get("name") or _t.get("abbreviation") or "").lower()
                    if _n:
                        _by_name[_n] = _row

                _patched = 0
                for _region in ["east", "west", "south", "midwest"]:
                    for _team in _raw.get(_region, []):
                        _row = _by_name.get((_team.get("name") or "").lower())
                        if not _row:
                            continue
                        _pace = _row.get("pace") or _row.get("possessions")
                        if _pace and 55 < float(_pace) < 85:
                            _team["pace"] = round(float(_pace), 1)
                            _patched += 1
                        _fg3a = _row.get("fg3a")
                        _fga = _row.get("fga")
                        if _fg3a and _fga and float(_fga) > 0:
                            _r = float(_fg3a) / float(_fga)
                            if 0.15 < _r < 0.60:
                                _team["three_pt_rate"] = round(_r, 3)
                                _patched += 1

                if _patched > 0:
                    with open(BRACKET_PATH, "w", encoding="utf-8") as _f:
                        _json.dump(_raw, _f, indent=2, ensure_ascii=False)
                    for _key in ("bracket_cache", "mc_cache"):
                        st.session_state.pop(_key, None)
                    st.success(f"Patched {_patched} stat fields. Reloading...")
                    st.rerun()
                else:
                    st.info("No stat updates found")
            except ValueError as _e:
                st.error(f"API key missing: {_e}")
            except Exception as _e:
                st.error(f"BDL error: {_e}")

# Mode indicator
mode_labels = {
    (0.0, 0.0): ("🏆", "CHALK", "Favorites win every game", "blue"),
    (0.1, 0.3): ("📊", "MODEL", "Probability-weighted outcomes", "green"),
    (0.4, 0.6): ("⚡", "STYLE-AWARE", "Style mismatches drive upsets", "orange"),
    (0.7, 0.8): ("🎭", "CINDERELLA", "Chaos favors Cinderella stories", "red"),
    (0.9, 1.0): ("🔥", "MAXIMUM CHAOS", "March Madness in full effect", "red"),
}
mode_info = ("📊", "MODEL", "Probability-weighted outcomes", "green")
for (lo, hi), info in mode_labels.items():
    if lo <= chaos_level <= hi:
        mode_info = info
        break
em, mn, md, mc = mode_info
if pool_optimal:
    st.success("**Pool Optimal** — 2× 12v5 + 1× 11v6 upset picks; all 1-seeds to Final Four")
else:
    st.info(f"**{em} {mn}** — {md}")

# ---------------------------------------------------------------------------
# Generate bracket (cached; pool_optimal included in cache key)
# ---------------------------------------------------------------------------
cache_key = f"bracket_{chaos_level}_{rng_seed}_{pool_optimal}"
if st.session_state.get("bracket_cache_key") != cache_key or "bracket_cache" not in st.session_state:
    if pool_optimal:
        with st.spinner("Building pool-optimal bracket..."):
            try:
                from backend.tournament.smart_bracket import generate_pool_optimal_bracket
                _po = generate_pool_optimal_bracket(bracket)
                region_rounds = {r: d["rounds"] for r, d in _po["regions"].items()}
                # Build FF + championship from region winners
                _ff = []
                for reg_a, reg_b in FF_PAIRINGS:
                    wa = _po["regions"][reg_a]["winner"]
                    wb = _po["regions"][reg_b]["winner"]
                    if wa and wb:
                        pa, _, _ = predict_game(wa, wb, 5)
                        if pa >= 0.5:
                            _w, _l, _p = wa, wb, pa
                        else:
                            _w, _l, _p = wb, wa, 1.0 - pa
                        _ff.append({"ta": wa, "tb": wb, "winner": _w, "loser": _l,
                                    "prob": _p, "is_upset": _w.seed > _l.seed,
                                    "regions": (reg_a, reg_b)})
                if len(_ff) == 2:
                    _ta, _tb = _ff[0]["winner"], _ff[1]["winner"]
                    _pc, _, _ = predict_game(_ta, _tb, 6)
                    if _pc >= 0.5:
                        _cw, _cl, _cp = _ta, _tb, _pc
                    else:
                        _cw, _cl, _cp = _tb, _ta, 1.0 - _pc
                    champ = {"ta": _ta, "tb": _tb, "winner": _cw, "loser": _cl,
                             "prob": _cp, "is_upset": _cw.seed > _cl.seed}
                else:
                    champ = {"ta": None, "tb": None, "winner": None,
                             "loser": None, "prob": 0.5, "is_upset": False}
                ff = _ff
                upsets_list = _po.get("upsets", [])
                pool_rationale = _po.get("pool_rationale", [])
            except Exception as e:
                st.error(f"Pool-optimal generator failed: {e}")
                st.stop()
    else:
        with st.spinner("Generating bracket predictions..."):
            region_rounds, ff, champ, upsets_list = generate_bracket(
                bracket, chaos_level=chaos_level, rng_seed=int(rng_seed)
            )
        pool_rationale = []
    _rationale = pool_rationale if pool_optimal else []
    st.session_state.bracket_cache = (region_rounds, ff, champ, upsets_list, _rationale)
    st.session_state.bracket_cache_key = cache_key

region_rounds, ff, champ, upsets_list, pool_rationale = st.session_state.bracket_cache

# Pool rationale callout (pool optimal mode only)
if pool_rationale:
    st.info(
        "**Why these upsets win pools:**\n\n"
        + "\n\n".join(f"• {r}" for r in pool_rationale)
    )

# ---------------------------------------------------------------------------
# Monte Carlo simulation (cached separately — only reruns when n_sims changes)
# ---------------------------------------------------------------------------
mc_probs = None
mc_cache_key = f"mc_{n_sims}_{rng_seed}"
if show_mc:
    if st.session_state.get("mc_cache_key") != mc_cache_key or "mc_cache" not in st.session_state:
        with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
            mc_probs_raw = run_quick_mc(bracket, n_sims=n_sims, chaos_seed=int(rng_seed))
        st.session_state.mc_cache = mc_probs_raw
        st.session_state.mc_cache_key = mc_cache_key
    mc_probs = st.session_state.mc_cache

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
st.subheader("Bracket Summary")

c1, c2, c3, c4 = st.columns(4)
champ_mc = (mc_probs or {}).get(champ["winner"].name, {}).get("champ") if (mc_probs and champ.get("winner")) else None
c1.metric(
    "🏆 Predicted Champion",
    f"#{champ['winner'].seed} {champ['winner'].name}" if champ.get("winner") else "TBD",
    f"{champ_mc:.1%} MC odds" if champ_mc else (f"{champ['prob']:.0%} in final" if champ.get("winner") else ""),
)
if ff:
    ff0_mc = (mc_probs or {}).get(ff[0]["winner"].name, {}).get("f4") if mc_probs else None
    c2.metric("F4 · Game 1", f"#{ff[0]['winner'].seed} {ff[0]['winner'].name}",
              f"{ff0_mc:.1%} MC F4 · {ff[0]['prob']:.0%} semi" if ff0_mc else f"{ff[0]['prob']:.0%} semi")
if len(ff) > 1:
    ff1_mc = (mc_probs or {}).get(ff[1]["winner"].name, {}).get("f4") if mc_probs else None
    c3.metric("F4 · Game 2", f"#{ff[1]['winner'].seed} {ff[1]['winner'].name}",
              f"{ff1_mc:.1%} MC F4 · {ff[1]['prob']:.0%} semi" if ff1_mc else f"{ff[1]['prob']:.0%} semi")
mode_label = "pool-optimal" if pool_optimal else f"chaos={chaos_level:.1f}"
c4.metric("⚡ Predicted Upsets", len(upsets_list), mode_label)

# ---------------------------------------------------------------------------
# Bracket HTML
# ---------------------------------------------------------------------------
st.subheader("📋 Full Bracket")
html = render_bracket_html(region_rounds, ff, champ, mc_probs=mc_probs,
                           chaos_level=chaos_level)
components.html(html, height=1080, scrolling=True)

st.divider()

# ---------------------------------------------------------------------------
# Key Insights panel
# ---------------------------------------------------------------------------
st.subheader("🔍 Key Insights")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**🎭 Top Cinderella Picks**")
    if mc_probs:
        cinderellas = [
            (name, probs["s16"], probs["f4"])
            for name, probs in mc_probs.items()
            if probs["f4"] >= 0.02
        ]
        # Get seed for each team
        team_seed = {t.name: t.seed for ts in bracket.values() for t in ts}
        cinderellas = [(n, s16, f4) for n, s16, f4 in cinderellas if team_seed.get(n, 1) >= 10]
        cinderellas.sort(key=lambda x: x[2], reverse=True)
        for name, s16, f4 in cinderellas[:5]:
            seed = team_seed.get(name, "?")
            st.markdown(f"#{seed} **{name}** · S16: {s16:.0%} · F4: {f4:.0%}")
    else:
        st.caption("Run with MC enabled to see Cinderella odds")

with col_b:
    st.markdown("**⚡ Most Dangerous First-Round Upsets**")
    # Compute R64 upset probabilities
    all_teams = {t.name: t for ts in bracket.values() for t in ts}
    upset_alerts = []
    for region, teams in bracket.items():
        seed_map = {t.seed: t for t in teams}
        for i in range(0, len(R64_SEED_ORDER), 2):
            s1, s2 = R64_SEED_ORDER[i], R64_SEED_ORDER[i + 1]
            if s1 not in seed_map or s2 not in seed_map:
                continue
            fav, dog = seed_map[s1], seed_map[s2]
            p, _, _ = predict_game(fav, dog, 1)
            upset_p = 1.0 - p
            if upset_p >= 0.30:
                upset_alerts.append((upset_p, fav, dog, region))
    upset_alerts.sort(key=lambda x: x[0], reverse=True)
    for upset_p, fav, dog, region in upset_alerts[:5]:
        st.markdown(f"#{dog.seed} **{dog.name}** over #{fav.seed} {fav.name} · {upset_p:.0%} · {region.title()}")
    if not upset_alerts:
        st.caption("No high-probability upsets (>30%) detected")

with col_c:
    st.markdown("**🏅 Champion Odds by Seed Tier**")
    if mc_probs:
        tier_probs = {"1-2 seeds": 0.0, "3-4 seeds": 0.0,
                      "5-8 seeds": 0.0, "9+ seeds": 0.0}
        team_seed = {t.name: t.seed for ts in bracket.values() for t in ts}
        for name, probs in mc_probs.items():
            seed = team_seed.get(name, 16)
            pct = probs["champ"]
            if seed <= 2: tier_probs["1-2 seeds"] += pct
            elif seed <= 4: tier_probs["3-4 seeds"] += pct
            elif seed <= 8: tier_probs["5-8 seeds"] += pct
            else: tier_probs["9+ seeds"] += pct
        for tier, pct in tier_probs.items():
            st.markdown(f"**{tier}**: {pct:.0%}")
    else:
        st.caption("Run with MC enabled")

st.divider()

# ---------------------------------------------------------------------------
# MC Probability Table
# ---------------------------------------------------------------------------
if mc_probs:
    with st.expander("📈 Full Monte Carlo Advancement Probabilities"):
        import pandas as pd
        team_seed = {t.name: t.seed for ts in bracket.values() for t in ts}
        team_region = {t.name: t.region for ts in bracket.values() for t in ts}
        rows = []
        for name, probs in mc_probs.items():
            rows.append({
                "Team": name,
                "Seed": team_seed.get(name, "?"),
                "Region": team_region.get(name, "?").title(),
                "R32": f"{probs['r32']:.0%}",
                "S16": f"{probs['s16']:.0%}",
                "E8":  f"{probs['e8']:.0%}",
                "F4":  f"{probs['f4']:.0%}",
                "Champion": f"{probs['champ']:.1%}",
                "_champ_sort": probs["champ"],
            })
        df = pd.DataFrame(rows).sort_values("_champ_sort", ascending=False).drop(columns=["_champ_sort"])
        st.dataframe(df, use_container_width=True, height=400)
        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "mc_probabilities.csv",
            "text/csv",
        )

# ---------------------------------------------------------------------------
# Upset list + round breakdown
# ---------------------------------------------------------------------------
if upsets_list:
    with st.expander(f"⚡ Upset Detail ({len(upsets_list)} predicted upsets)"):
        round_names = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Championship"}
        for u in upsets_list:
            st.markdown(
                f"**{u['region'].upper()} · {round_names.get(u['round'], str(u['round']))}** — "
                f"#{u['winner_seed']} **{u['winner']}** over #{u['loser_seed']} {u['loser']} "
                f"({u['prob']:.0%})"
            )

with st.expander("Round-by-Round Text Breakdown"):
    round_names = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8"}
    for region in REGIONS:
        st.markdown(f"**{region.upper()}**")
        for rnd in [1, 2, 3, 4]:
            st.markdown(f"*{round_names[rnd]}*")
            for m in region_rounds[region][rnd]:
                tag = " ⚡ UPSET" if m["is_upset"] else ""
                st.markdown(
                    f"  #{m['ta'].seed} {m['ta'].name} vs #{m['tb'].seed} {m['tb'].name} "
                    f"→ **#{m['winner'].seed} {m['winner'].name}** ({m['prob']:.0%}){tag}"
                )
        st.markdown("---")
    st.markdown("**FINAL FOUR**")
    for m in ff:
        st.markdown(
            f"#{m['ta'].seed} {m['ta'].name} ({m['regions'][0].title()}) "
            f"vs #{m['tb'].seed} {m['tb'].name} ({m['regions'][1].title()}) "
            f"→ **#{m['winner'].seed} {m['winner'].name}** ({m['prob']:.0%})"
        )
    st.markdown("**CHAMPIONSHIP**")
    if champ["ta"] and champ["tb"]:
        st.markdown(
            f"#{champ['ta'].seed} {champ['ta'].name} vs "
            f"#{champ['tb'].seed} {champ['tb'].name} "
            f"→ **#{champ['winner'].seed} {champ['winner'].name}** ({champ['prob']:.0%})"
        )
