"""
Predicted Bracket Visual — full 68-team bracket with model-projected winners.

Shows the "chalk" bracket: each game is won by whichever team the V9.1 model
gives a higher win probability. Win probability is shown on each matchup.

Layout:
    East   (top-left)  advances right  ──────────┐
                                                   ├─ F4 Left  ─┐
    South  (bot-right) advances left  ──────────┘               │
                                                                 ├── CHAMPION
    West   (top-right) advances left  ──────────┐               │
                                                   ├─ F4 Right ─┘
    Midwest (bot-left) advances right ──────────┘

F4 pairings: South vs East · West vs Midwest
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

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
    from backend.tournament.bracket_simulator import R64_SEED_ORDER
    TOURNAMENT_OK = True
except ImportError as e:
    TOURNAMENT_OK = False
    st.error(f"Tournament module not available: {e}")
    st.stop()

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bracket_2026.json"
REGIONS = ["east", "south", "west", "midwest"]
FF_PAIRINGS = [("south", "east"), ("west", "midwest")]


# ---------------------------------------------------------------------------
# Bracket loading helpers (same as page 13 but standalone)
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
                composite_rating=t.get("composite_rating", 0.0),
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


def load_bracket_from_session() -> dict | None:
    if "bracket_teams" not in st.session_state:
        return None
    bracket = {}
    for region in REGIONS:
        entries = st.session_state.bracket_teams.get(region, [])
        if not entries or not any(e.get("name", "").strip() for e in entries):
            return None
        teams = []
        for e in entries:
            name = e.get("name", "").strip()
            if not name:
                name = f"{region.title()}-Seed{e['seed']}"
            teams.append(TournamentTeam(
                name=name, seed=e["seed"], region=region,
                composite_rating=e.get("composite_rating", 0.0),
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
# Predicted bracket generator
# ---------------------------------------------------------------------------
def generate_predicted_bracket(bracket: dict) -> tuple:
    """
    Pick the higher-probability team in every matchup to produce the 'chalk' bracket.

    Returns:
        region_rounds: {region: {round_num: [matchup_dict, ...]}}
            where matchup_dict = {ta, tb, winner, loser, prob, is_upset}
        ff: list of 2 F4 matchup dicts
        champ: championship matchup dict
    """
    region_rounds = {}

    for region, teams in bracket.items():
        seed_to_team = {t.seed: t for t in teams}
        slots = [seed_to_team[s] for s in R64_SEED_ORDER if s in seed_to_team]

        rounds = {}
        # Round 0: initial R64 pairs (just team objects, not yet played)
        rounds[0] = [(slots[i], slots[i + 1]) for i in range(0, len(slots), 2)]

        current = slots
        for round_num in [1, 2, 3, 4]:
            matchups = []
            next_round = []
            for i in range(0, len(current), 2):
                ta, tb = current[i], current[i + 1]
                prob_a, _, _ = predict_game(ta, tb, round_num)
                if prob_a >= 0.5:
                    winner, loser, prob = ta, tb, prob_a
                else:
                    winner, loser, prob = tb, ta, 1.0 - prob_a
                is_upset = winner.seed > loser.seed
                matchups.append({
                    "ta": ta, "tb": tb,
                    "winner": winner, "loser": loser,
                    "prob": prob, "is_upset": is_upset,
                })
                next_round.append(winner)
            rounds[round_num] = matchups
            current = next_round

        region_rounds[region] = rounds

    # Final Four
    ff = []
    for reg_a, reg_b in FF_PAIRINGS:
        ta = region_rounds[reg_a][4][0]["winner"]
        tb = region_rounds[reg_b][4][0]["winner"]
        prob_a, _, _ = predict_game(ta, tb, 5)
        if prob_a >= 0.5:
            winner, loser, prob = ta, tb, prob_a
        else:
            winner, loser, prob = tb, ta, 1.0 - prob_a
        ff.append({
            "ta": ta, "tb": tb, "winner": winner, "loser": loser,
            "prob": prob, "is_upset": winner.seed > loser.seed,
            "regions": (reg_a, reg_b),
        })

    # Championship
    ta, tb = ff[0]["winner"], ff[1]["winner"]
    prob_a, _, _ = predict_game(ta, tb, 6)
    if prob_a >= 0.5:
        winner, loser, prob = ta, tb, prob_a
    else:
        winner, loser, prob = tb, ta, 1.0 - prob_a
    champ = {
        "ta": ta, "tb": tb, "winner": winner, "loser": loser,
        "prob": prob, "is_upset": winner.seed > loser.seed,
    }

    return region_rounds, ff, champ


# ---------------------------------------------------------------------------
# HTML bracket renderer
# ---------------------------------------------------------------------------
TEAM_W = 138
TEAM_H = 24
ROUND_GAP = 6

SEED_COLORS = {
    (1, 2):   "#1565C0",   # dark blue  — top seeds
    (3, 4):   "#2E7D32",   # dark green
    (5, 8):   "#BF360C",   # deep orange
    (9, 12):  "#6A1B9A",   # purple
    (13, 16): "#4E342E",   # brown
}

def _seed_color(seed: int) -> str:
    for (lo, hi), color in SEED_COLORS.items():
        if lo <= seed <= hi:
            return color
    return "#424242"


def _team_box(team: TournamentTeam, is_winner: bool, win_prob: float | None = None,
              is_champ: bool = False) -> str:
    color = _seed_color(team.seed)
    if is_champ:
        bg = "#FFF9C4"; border = "2px solid #F9A825"; fw = "bold"
    elif is_winner:
        bg = "#E8F5E9"; border = "1px solid #43A047"; fw = "bold"
    else:
        bg = "#F5F5F5"; border = "1px solid #BDBDBD"; fw = "normal"
    opacity = "1" if is_winner else "0.55"
    prob_html = (f"<span style='font-size:9px;color:#1565C0;margin-left:3px'>"
                 f"{win_prob:.0%}</span>") if win_prob and is_winner else ""
    return (
        f'<div style="width:{TEAM_W}px;height:{TEAM_H}px;border:{border};background:{bg};'
        f'display:flex;align-items:center;padding:0 4px;box-sizing:border-box;'
        f'font-size:11px;opacity:{opacity};font-family:\'Segoe UI\',Arial,sans-serif;">'
        f'<span style="color:{color};font-weight:bold;min-width:18px;font-size:10px">#{team.seed}</span>'
        f'<span style="font-weight:{fw};overflow:hidden;text-overflow:ellipsis;'
        f'white-space:nowrap;flex:1;margin-left:2px;font-size:11px">{team.name}</span>'
        f'{prob_html}'
        f'</div>'
    )


def _matchup_block(matchup: dict, margin_px: int, show_prob_on: str = "winner") -> str:
    """Render one two-team matchup block with top/bottom margin for bracket spacing."""
    ta, tb = matchup["ta"], matchup["tb"]
    winner = matchup["winner"]
    prob = matchup["prob"]
    a_wins = winner is ta
    ta_html = _team_box(ta, a_wins, prob if a_wins else None)
    tb_html = _team_box(tb, not a_wins, prob if not a_wins else None)
    return (
        f'<div style="margin:{margin_px}px 0">'
        f'{ta_html}{tb_html}'
        f'</div>'
    )


def _round_col(matchups: list, round_num: int, gap_px: int = ROUND_GAP,
               label: str = "") -> str:
    """Render one round column with correct bracket vertical spacing."""
    margin = (2 ** (round_num - 1) - 1) * TEAM_H
    label_html = (f'<div style="text-align:center;font-size:10px;font-weight:bold;'
                  f'color:#555;margin-bottom:4px;font-family:Arial">{label}</div>') if label else ""
    blocks = "".join(_matchup_block(m, margin) for m in matchups)
    return (
        f'<div style="display:flex;flex-direction:column;margin:0 {gap_px}px">'
        f'{label_html}{blocks}</div>'
    )


def _region_bracket_ltr(region: str, region_rounds: dict, label: str = "") -> str:
    """One region bracket advancing left-to-right (East, Midwest)."""
    label_html = (
        f'<div style="text-align:center;font-size:12px;font-weight:bold;color:#1565C0;'
        f'margin-bottom:6px;font-family:Arial;letter-spacing:0.5px">{label}</div>'
    ) if label else ""

    r64_col = _r64_col(region, region_rounds)
    r32_col  = _round_col(region_rounds[region][2], 2, label="R32")
    s16_col  = _round_col(region_rounds[region][3], 3, label="S16")
    e8_col   = _round_col(region_rounds[region][4], 4, label="E8")

    return (
        f'<div>'
        f'{label_html}'
        f'<div style="display:flex;flex-direction:row;align-items:flex-start">'
        f'{r64_col}{r32_col}{s16_col}{e8_col}'
        f'</div></div>'
    )


def _region_bracket_rtl(region: str, region_rounds: dict, label: str = "") -> str:
    """One region bracket advancing right-to-left (West, South) — columns reversed."""
    label_html = (
        f'<div style="text-align:center;font-size:12px;font-weight:bold;color:#1565C0;'
        f'margin-bottom:6px;font-family:Arial;letter-spacing:0.5px">{label}</div>'
    ) if label else ""

    r64_col = _r64_col(region, region_rounds)
    r32_col  = _round_col(region_rounds[region][2], 2, label="R32")
    s16_col  = _round_col(region_rounds[region][3], 3, label="S16")
    e8_col   = _round_col(region_rounds[region][4], 4, label="E8")

    # Reversed order: E8 leftmost, R64 rightmost
    return (
        f'<div>'
        f'{label_html}'
        f'<div style="display:flex;flex-direction:row;align-items:flex-start">'
        f'{e8_col}{s16_col}{r32_col}{r64_col}'
        f'</div></div>'
    )


def _r64_col(region: str, region_rounds: dict) -> str:
    """R64 column — shows all 16 teams grouped as 8 matchups."""
    pairs = region_rounds[region][0]  # list of (ta, tb) tuples
    matchups_r1 = region_rounds[region][1]  # R64 results with winner info

    blocks = []
    for idx, ((ta, tb), m) in enumerate(zip(pairs, matchups_r1)):
        a_wins = m["winner"] is ta
        ta_html = _team_box(ta, a_wins, m["prob"] if a_wins else None)
        tb_html = _team_box(tb, not a_wins, m["prob"] if not a_wins else None)
        blocks.append(f'<div style="margin:0">{ta_html}{tb_html}</div>')

    label_html = (
        f'<div style="text-align:center;font-size:10px;font-weight:bold;'
        f'color:#555;margin-bottom:4px;font-family:Arial">R64</div>'
    )
    return (
        f'<div style="display:flex;flex-direction:column;margin:0 {ROUND_GAP}px">'
        f'{label_html}{"".join(blocks)}</div>'
    )


def _ff_center(ff: list, champ: dict) -> str:
    """Final Four + Championship center column."""
    total_h = 16 * TEAM_H * 2  # height of both regions (East+South or Midwest+West)

    def ff_slot(m: dict, label: str) -> str:
        ta, tb = m["ta"], m["tb"]
        a_wins = m["winner"] is ta
        ta_html = _team_box(ta, a_wins, m["prob"] if a_wins else None)
        tb_html = _team_box(tb, not a_wins, m["prob"] if not a_wins else None)
        lbl = (f'<div style="text-align:center;font-size:10px;font-weight:bold;'
               f'color:#E65100;margin-bottom:4px;font-family:Arial">{label}</div>')
        return f'<div style="margin-bottom:8px">{lbl}{ta_html}{tb_html}</div>'

    champion_html = _team_box(champ["winner"], True, champ["prob"], is_champ=True)
    runner_up_html = _team_box(champ["loser"], False)
    champ_lbl = (
        f'<div style="text-align:center;font-size:11px;font-weight:bold;'
        f'color:#B71C1C;margin-bottom:4px;font-family:Arial">CHAMPION</div>'
    )
    runner_up_lbl = (
        f'<div style="text-align:center;font-size:10px;color:#555;'
        f'margin-bottom:2px;font-family:Arial">Runner-Up</div>'
    )

    ff0 = ff_slot(ff[0], f"F4: {ff[0]['regions'][0].title()} vs {ff[0]['regions'][1].title()}")
    ff1 = ff_slot(ff[1], f"F4: {ff[1]['regions'][0].title()} vs {ff[1]['regions'][1].title()}")

    inner = (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'justify-content:center;height:100%">'
        f'{ff0}'
        f'<div style="height:16px"></div>'
        f'{champ_lbl}{champion_html}'
        f'<div style="height:6px"></div>'
        f'{runner_up_lbl}{runner_up_html}'
        f'<div style="height:16px"></div>'
        f'{ff1}'
        f'</div>'
    )

    return (
        f'<div style="width:{TEAM_W + 24}px;display:flex;flex-direction:column;'
        f'justify-content:center;padding:0 {ROUND_GAP}px;border-left:2px dashed #BDBDBD;'
        f'border-right:2px dashed #BDBDBD;">'
        f'{inner}</div>'
    )


def render_full_bracket_html(region_rounds: dict, ff: list, champ: dict) -> str:
    """Render the full 4-region bracket as self-contained HTML."""

    east    = _region_bracket_ltr("east",    region_rounds, "EAST")
    midwest = _region_bracket_ltr("midwest", region_rounds, "MIDWEST")
    west    = _region_bracket_rtl("west",    region_rounds, "WEST")
    south   = _region_bracket_rtl("south",   region_rounds, "SOUTH")
    center  = _ff_center(ff, champ)

    # Layout:
    #  Row 1: East (L-to-R) | Center | West (R-to-L)
    #  Row 2: Midwest (L-to-R) |       | South (R-to-L)
    row1 = (
        f'<div style="display:flex;flex-direction:row;align-items:flex-start;margin-bottom:16px">'
        f'{east}'
        f'{center}'
        f'{west}'
        f'</div>'
    )
    row2 = (
        f'<div style="display:flex;flex-direction:row;align-items:flex-start">'
        f'{midwest}'
        f'<div style="width:{TEAM_W + 24 + 2*ROUND_GAP}px"></div>'
        f'{south}'
        f'</div>'
    )

    bracket_label = (
        f'<div style="text-align:center;font-size:16px;font-weight:bold;color:#212121;'
        f'margin-bottom:12px;font-family:Arial;letter-spacing:1px">'
        f'2026 NCAA TOURNAMENT — MODEL PREDICTED BRACKET'
        f'<span style="font-size:11px;font-weight:normal;color:#666;margin-left:12px">'
        f'(V9.1 model · chalk picks · win % shown on winner)'
        f'</span></div>'
    )

    legend = (
        f'<div style="display:flex;gap:16px;margin-bottom:12px;font-size:10px;font-family:Arial">'
        + "".join(
            f'<span><span style="background:{c};color:white;padding:1px 5px;border-radius:2px">'
            f'#{lo}-{hi}</span> seed</span>'
            for (lo, hi), c in SEED_COLORS.items()
        )
        + f'<span style="margin-left:8px">&#9654; Bold = predicted winner</span>'
        + f'<span style="background:#FFF9C4;border:2px solid #F9A825;padding:1px 5px">Champion</span>'
        + f'</div>'
    )

    return f"""
    <html><head><meta charset="utf-8"></head>
    <body style="margin:0;padding:12px;background:#FAFAFA">
    {bracket_label}
    {legend}
    <div style="overflow-x:auto;overflow-y:auto">
        {row1}
        {row2}
    </div>
    </body></html>
    """


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Predicted Tournament Bracket")
st.caption("Model picks the higher-probability team in every matchup. Bracket uses V9.1 composite ratings + historical seed blending.")

# Load bracket — prefer session state from page 13, fall back to disk
bracket = load_bracket_from_session()
source = "Tournament Bracket page (page 13)"
if bracket is None:
    try:
        bracket = load_bracket_from_disk()
        source = "data/bracket_2026.json (pre-built)"
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Go to page 13 (Tournament Bracket) and click 'Load 2026 Bracket' first.")
        st.stop()

named = sum(1 for ts in bracket.values() for t in ts if not t.name.startswith(f"{t.region.title()}-Seed"))
st.caption(f"Bracket source: {source} | {named}/64 teams named")

# Re-generate button
if st.button("Generate / Refresh Predicted Bracket", type="primary"):
    st.session_state.pop("predicted_bracket", None)

if "predicted_bracket" not in st.session_state:
    with st.spinner("Computing predicted bracket..."):
        region_rounds, ff, champ = generate_predicted_bracket(bracket)
        st.session_state.predicted_bracket = (region_rounds, ff, champ)

region_rounds, ff, champ = st.session_state.predicted_bracket

# ---------------------------------------------------------------------------
# Key callouts
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Champion", f"#{champ['winner'].seed} {champ['winner'].name}",
          f"{champ['prob']:.0%} in final")
c2.metric("Final Four Game 1",
          f"#{ff[0]['winner'].seed} {ff[0]['winner'].name}",
          f"{ff[0]['prob']:.0%} | {ff[0]['regions'][0].title()} vs {ff[0]['regions'][1].title()}")
c3.metric("Final Four Game 2",
          f"#{ff[1]['winner'].seed} {ff[1]['winner'].name}",
          f"{ff[1]['prob']:.0%} | {ff[1]['regions'][0].title()} vs {ff[1]['regions'][1].title()}")
upsets = sum(
    1 for rr in region_rounds.values()
    for rnd in [1, 2, 3, 4]
    for m in rr[rnd]
    if m["is_upset"]
)
c4.metric("Predicted Upsets", upsets, "in rounds 1-4")

st.divider()

# ---------------------------------------------------------------------------
# Full bracket visual
# ---------------------------------------------------------------------------
html = render_full_bracket_html(region_rounds, ff, champ)
# Height: 16 teams * 24px * 2 regions + padding
components.html(html, height=1050, scrolling=True)

st.divider()

# ---------------------------------------------------------------------------
# Round-by-round text summary (expandable)
# ---------------------------------------------------------------------------
with st.expander("Round-by-round breakdown (text)"):
    round_names = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8"}
    for region in ["east", "south", "west", "midwest"]:
        st.markdown(f"**{region.upper()} REGION**")
        for rnd in [1, 2, 3, 4]:
            matchups = region_rounds[region][rnd]
            st.markdown(f"*{round_names[rnd]}*")
            for m in matchups:
                upset_tag = " **UPSET**" if m["is_upset"] else ""
                st.markdown(
                    f"  #{m['ta'].seed} {m['ta'].name} vs #{m['tb'].seed} {m['tb'].name} "
                    f"→ **#{m['winner'].seed} {m['winner'].name}** ({m['prob']:.0%}){upset_tag}"
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
    st.markdown(
        f"#{champ['ta'].seed} {champ['ta'].name} vs "
        f"#{champ['tb'].seed} {champ['tb'].name} "
        f"→ **#{champ['winner'].seed} {champ['winner'].name}** ({champ['prob']:.0%})"
    )
