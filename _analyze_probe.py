import json, os, glob
from collections import Counter, defaultdict

out_dir = 'postman_collections/responses'
files = {os.path.basename(f): f for f in glob.glob(os.path.join(out_dir, '*20260422_101629*.json'))}

def load_json(fname):
    fpath = files.get(fname)
    if not fpath:
        return None
    with open(fpath, encoding='utf-8') as fp:
        return json.load(fp)

print('='*70)
print('ROSTER ANALYSIS')
print('='*70)
roster = load_json('api_fantasy_roster_20260422_101629.json')
players = roster.get('players', [])
print(f'Total players: {len(players)}')

null_counts = defaultdict(int)
for p in players:
    for k, v in p.items():
        if v is None:
            null_counts[k] += 1

for k, v in sorted(null_counts.items(), key=lambda x: -x[1]):
    print(f'  {k} null: {v}/{len(players)}')

for window in ['rolling_7d', 'rolling_14d', 'rolling_15d', 'rolling_30d']:
    present = sum(1 for p in players if p.get(window) is not None)
    print(f'  {window} present: {present}/{len(players)}')

bdl_null = sum(1 for p in players if p.get('bdl_player_id') is None)
mlbam_null = sum(1 for p in players if p.get('mlbam_id') is None)
own_zero = sum(1 for p in players if p.get('ownership_pct') == 0.0)
print(f'  bdl_player_id null: {bdl_null}/{len(players)}')
print(f'  mlbam_id null: {mlbam_null}/{len(players)}')
print(f'  ownership_pct == 0.0: {own_zero}/{len(players)}')

fetched_null = sum(1 for p in players if p.get('freshness', {}).get('fetched_at') is None)
print(f'  freshness.fetched_at null: {fetched_null}/{len(players)}')

ss_nulls = defaultdict(int)
all_null_players = []
for p in players:
    ss = p.get('season_stats', {})
    vals = ss.get('values', {})
    all_null = all(v is None for v in vals.values()) if vals else True
    if all_null and vals:
        all_null_players.append(p['player_name'])
    for k, v in vals.items():
        if v is None:
            ss_nulls[k] += 1

print('  Season stats completely null for:', all_null_players)
print('  season_stats nulls by field:')
for k, v in sorted(ss_nulls.items(), key=lambda x: -x[1]):
    print(f'    {k}: {v}/{len(players)}')

print('  Schema checks:')
for p in players:
    pos = p.get('eligible_positions', [])
    ss = p.get('season_stats', {}).get('values', {})
    is_pitcher = any(x in pos for x in ['SP', 'RP', 'P'])
    if is_pitcher and ss.get('ERA') is None and ss.get('IP') is None and ss.get('W') is None:
        if any(v is not None for v in ss.values()):
            pass
        elif ss:
            print(f'    PITCHER all stats null: {p["player_name"]} {pos}')
    if not is_pitcher and ss.get('AVG') is None and ss.get('H') is None and ss.get('R') is None:
        if any(v is not None for v in ss.values()):
            pass
        elif ss:
            print(f'    BATTER all stats null: {p["player_name"]} {pos}')

for p in players:
    pos = p.get('eligible_positions', [])
    ss = p.get('season_stats', {}).get('values', {})
    is_pitcher = any(x in pos for x in ['SP', 'RP', 'P'])
    if not is_pitcher:
        if ss.get('ERA') is not None or ss.get('WHIP') is not None or ss.get('W') is not None:
            print(f'    BATTER with pitcher stats: {p["player_name"]} ERA={ss.get("ERA")} WHIP={ss.get("WHIP")} W={ss.get("W")}')

print()
print('='*70)
print('LINEUP ANALYSIS')
print('='*70)
lineup = load_json('api_fantasy_lineup_2026_04_22_20260422_101629.json')
print('Keys:', list(lineup.keys()))
print('games_count:', lineup.get('games_count'))
print('date:', lineup.get('date'))
print('no_games_today:', lineup.get('no_games_today'))
print('lineup_warnings:', lineup.get('lineup_warnings'))

batters = lineup.get('batters', [])
pitchers = lineup.get('pitchers', [])
print(f'Batters: {len(batters)}, Pitchers: {len(pitchers)}')

for role, plist in [('Batter', batters), ('Pitcher', pitchers)]:
    for key in ['lineup_status', 'eligible_positions', 'has_game', 'bdl_player_id', 'mlbam_id', 'game_time', 'weather', 'opponent']:
        null_count = sum(1 for p in plist if p.get(key) is None)
        print(f'  {role} {key} null: {null_count}/{len(plist)}')
    if role == 'Pitcher':
        null_count = sum(1 for p in plist if p.get('is_two_start') is None)
        print(f'  {role} is_two_start null: {null_count}/{len(plist)}')

has_game_b = sum(1 for b in batters if b.get('has_game'))
has_game_p = sum(1 for p in pitchers if p.get('has_game'))
print(f'  Batters has_game=True: {has_game_b}/{len(batters)}')
print(f'  Pitchers has_game=True: {has_game_p}/{len(pitchers)}')

if batters:
    print('  Sample batter keys:', list(batters[0].keys()))
if pitchers:
    print('  Sample pitcher keys:', list(pitchers[0].keys()))

print()
print('='*70)
print('WAIVER ANALYSIS')
print('='*70)
waiver = load_json('api_fantasy_waiver_position_ALL_player_type_ALL_20260422_101629.json')
available = waiver.get('top_available', [])
print(f'Top available: {len(available)}')

for key in ['owned_pct', 'starts_this_week', 'statcast_signals', 'hot_cold', 'status', 'injury_note', 'injury_status', 'statcast_stats']:
    if key == 'owned_pct':
        empty = sum(1 for p in available if p.get(key) == 0.0)
    elif key == 'starts_this_week':
        empty = sum(1 for p in available if p.get(key) == 0)
    elif key == 'statcast_signals':
        empty = sum(1 for p in available if p.get(key) == [])
    else:
        empty = sum(1 for p in available if p.get(key) is None)
    print(f'  {key} empty/null: {empty}/{len(available)}')

cc_empty = sum(1 for p in available if p.get('category_contributions') == {})
print(f'  category_contributions empty: {cc_empty}/{len(available)}')

non_empty_cc = [p for p in available if p.get('category_contributions') != {}]
print(f'  Non-empty category_contributions: {len(non_empty_cc)}')
for p in non_empty_cc:
    print(f'    {p["name"]}: {p["category_contributions"]}')

print('  ERA=0.00 check:')
for p in available:
    stats = p.get('stats', {})
    era = stats.get('ERA')
    ip = stats.get('IP')
    if era == '0.00' and ip and float(ip) > 0:
        print(f'    {p["name"]}: ERA=0.00 with IP={ip}')

print('  Batter with pitcher stats:')
for p in available:
    pos = p.get('position', '')
    stats = p.get('stats', {})
    if pos in ('1B', '2B', '3B', 'SS', 'C', 'CF', 'LF', 'RF', 'OF', 'Util', 'DH'):
        if 'IP' in stats or 'ERA' in stats or 'WHIP' in stats:
            print(f'    {p["name"]} ({pos}): {stats}')

print('  Pitcher with batter stats:')
for p in available:
    pos = p.get('position', '')
    stats = p.get('stats', {})
    if pos in ('SP', 'RP'):
        if 'HR_B' in stats or 'RBI' in stats or 'AVG' in stats:
            print(f'    {p["name"]} ({pos}): {stats}')

print()
print('='*70)
print('DECISIONS ANALYSIS')
print('='*70)
decisions = load_json('api_fantasy_decisions_20260422_101629.json')
decs = decisions.get('decisions', [])
lineup_decs = [d for d in decs if d['decision']['decision_type'] == 'lineup']
waiver_decs = [d for d in decs if d['decision']['decision_type'] == 'waiver']
print(f'Total: {len(decs)}, Lineup: {len(lineup_decs)}, Waiver: {len(waiver_decs)}')

as_of_dates = set(d['decision']['as_of_date'] for d in decs)
print(f'as_of_date values: {as_of_dates}')

vg_null = sum(1 for d in decs if d['decision'].get('value_gain') is None)
rn_null = sum(1 for d in decs if d.get('explanation', {}).get('risk_narrative') is None)
tr_null = sum(1 for d in decs if d.get('explanation', {}).get('track_record_narrative') is None)
print(f'value_gain null: {vg_null}/{len(decs)}')
print(f'risk_narrative null: {rn_null}/{len(decs)}')
print(f'track_record_narrative null: {tr_null}/{len(decs)}')

# Drop targets
drop_targets = Counter()
for d in waiver_decs:
    drop_targets[d['decision'].get('drop_player_name', 'NONE')] += 1
print(f'Drop target distribution: {dict(drop_targets)}')

# Impossible projections
impossible = []
for d in decs:
    exp = d.get('explanation', {})
    summary = exp.get('summary', '')
    if '0.00 ERA' in summary or '0.00 WHIP' in summary:
        impossible.append((d['decision']['player_name'], summary))
print(f'Impossible projections: {len(impossible)}')
for name, summary in impossible:
    print(f'  {name}: {summary[:120]}')

print()
print('='*70)
print('MATCHUP ANALYSIS')
print('='*70)
matchup = load_json('api_fantasy_matchup_20260422_101629.json')
print('Keys:', list(matchup.keys()))
print('week:', matchup.get('week'))
print('is_playoffs:', matchup.get('is_playoffs'))
print('message:', matchup.get('message'))
my_stats = matchup.get('my_team', {}).get('stats', {})
opp_stats = matchup.get('opponent', {}).get('stats', {})
print('My stats:', my_stats)
print('Opp stats:', opp_stats)

# Check for illogical values
for team_name, stats in [('My', my_stats), ('Opp', opp_stats)]:
    for k, v in stats.items():
        if v in ('', None):
            print(f'  {team_name} {k} is empty/null')
        if k == 'NSB' and isinstance(v, str):
            try:
                nv = int(v)
                if nv < -20 or nv > 20:
                    print(f'  {team_name} NSB suspicious: {v}')
            except:
                pass
        if k in ('ERA', 'WHIP') and isinstance(v, str):
            try:
                fv = float(v)
                if fv < 0:
                    print(f'  {team_name} {k} negative: {v}')
                if fv == 0.0:
                    print(f'  {team_name} {k} is 0.00')
            except:
                pass

print()
print('='*70)
print('BRIEFING ANALYSIS')
print('='*70)
briefing = load_json('api_fantasy_briefing_2026_04_22_20260422_101629.json')
print('date:', briefing.get('date'))
cats = briefing.get('categories', [])
print('categories count:', len(cats))
print('starters:', len(briefing.get('starters', [])))
print('bench:', len(briefing.get('bench', [])))
print('two_start_pitchers:', len(briefing.get('two_start_pitchers', [])))
print('Category names:')
for c in cats:
    print(f'  name={c.get("name")!r} rank={c.get("rank")} value={c.get("value")}')

print()
print('='*70)
print('DRAFT BOARD ANALYSIS')
print('='*70)
draft = load_json('api_fantasy_draft_board_limit_200_20260422_101629.json')
players = draft.get('players', [])
print(f'Total: {len(players)}')
age_zero = sum(1 for p in players if p.get('age') == 0)
print(f'age == 0: {age_zero}/{len(players)}')

print()
print('='*70)
print('PIPELINE ANALYSIS')
print('='*70)
pipe = load_json('admin_pipeline_health_20260422_101629.json')
print('Type:', type(pipe))
if isinstance(pipe, dict):
    print('Keys:', list(pipe.keys()))
    for k, v in pipe.items():
        if k != 'tables':
            print(f'  {k}: {v}')
    tables = pipe.get('tables', {})
    if isinstance(tables, list):
        for t in tables:
            print(f'  {t["name"]}: healthy={t.get("healthy")} rows={t.get("row_count")} latest={t.get("latest_date")}')
    elif isinstance(tables, dict):
        for k, v in tables.items():
            print(f'  {k}: rows={v.get("row_count")} latest={v.get("latest_date")}')

print()
print('='*70)
print('SCHEDULER ANALYSIS')
print('='*70)
sched = load_json('admin_scheduler_status_20260422_101629.json')
print('running:', sched.get('running'))
print('job_count:', len(sched.get('jobs', [])))
for j in sched.get('jobs', []):
    print(f'  {j.get("id")}: name={j.get("name")} next_run={j.get("next_run_time")}')

print()
print('='*70)
print('ERROR RESPONSES')
print('='*70)
for fname in ['api_fantasy_waiver_recommendations_20260422_101629.json',
              'api_fantasy_player_scores_period_season_20260422_101629.json',
              'admin_validate_system_20260422_101629.json',
              'api_fantasy_roster_optimize_20260422_101629.json',
              'api_fantasy_matchup_simulate_20260422_101629.json']:
    data = load_json(fname)
    print(f'{fname}: {data}')
