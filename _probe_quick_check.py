import json

files = {
    'lineup': 'postman_collections/responses/api_fantasy_lineup_2026_04_20_20260420_181810.json',
    'briefing': 'postman_collections/responses/api_fantasy_briefing_2026_04_20_20260420_181810.json',
    'matchup': 'postman_collections/responses/api_fantasy_matchup_20260420_181810.json',
    'pipeline': 'postman_collections/responses/admin_pipeline_health_20260420_181810.json',
    'scheduler': 'postman_collections/responses/admin_scheduler_status_20260420_181810.json',
    'decisions': 'postman_collections/responses/api_fantasy_decisions_20260420_181810.json',
    'waiver': 'postman_collections/responses/api_fantasy_waiver_position_ALL_player_type_ALL_20260420_181810.json',
}

for name, path in files.items():
    try:
        d = json.load(open(path))
        if name == 'lineup':
            print(f"{name}: games_count={d.get('games_count')}, no_games_today={d.get('no_games_today')}, batters={len(d.get('batters',[]))}, pitchers={len(d.get('pitchers',[]))}")
        elif name == 'briefing':
            print(f"{name}: categories={len(d.get('categories',[]))}, starters={len(d.get('starters',[]))}, bench={len(d.get('bench',[]))}")
        elif name == 'matchup':
            my_stats = d.get('my_team',{}).get('stats',{})
            opp_stats = d.get('opponent',{}).get('stats',{})
            print(f"{name}: my_keys={len(my_stats)}, opp_keys={len(opp_stats)}")
        elif name == 'pipeline':
            print(f"{name}: overall_healthy={d.get('overall_healthy')}, tables={len(d.get('tables',[]))}")
            for t in d.get('tables',[]):
                print(f"  - {t['name']}: rows={t['row_count']}, latest={t['latest_date']}")
        elif name == 'scheduler':
            print(f"{name}: jobs={len(d.get('jobs',[]))}")
        elif name == 'decisions':
            decs = d.get('decisions',[])
            lineup = [x for x in decs if x['decision']['decision_type']=='lineup']
            waiver = [x for x in decs if x['decision']['decision_type']=='waiver']
            print(f"{name}: total={len(decs)}, lineup={len(lineup)}, waiver={len(waiver)}")
        elif name == 'waiver':
            avail = d.get('top_available',[])
            print(f"{name}: top_available={len(avail)}, opponent={d.get('matchup_opponent')}, closer_alert={d.get('closer_alert')}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
