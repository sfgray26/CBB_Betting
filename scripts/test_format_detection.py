"""
Test the Yahoo API format detection logic locally.
"""

# Simulate the roster_wrapper structure from the actual Yahoo API response
roster_wrapper = {
    'coverage_type': 'all',
    'date': '2026-04-09',
    'is_prescoring': False,
    'is_editable': True,
    '0': {  # This is a player entry
        'player_key': '469.p.12345',
        'name': 'Test Player'
    },
    'outs_pitched': 100
}

print("Testing Yahoo API format detection logic")
print("=" * 70)
print()

print(f"roster_wrapper type: {type(roster_wrapper).__name__}")
print(f"roster_wrapper keys: {list(roster_wrapper.keys())}")
print()

# Test the old logic (before fix)
print("OLD LOGIC:")
players_raw = roster_wrapper.get("players", {})
print(f"players_raw: {players_raw}")
print(f"len(players_raw): {len(players_raw)}")
print(f"bool(players_raw): {bool(players_raw)}")
print(f"Evaluates to: {not players_raw or (isinstance(players_raw, dict) and len(players_raw) == 0)}")
print()

# Test the new logic (after fix)
print("NEW LOGIC:")
has_players_key = "players" in roster_wrapper
players_raw = roster_wrapper.get("players", {})
players_raw_has_data = len(players_raw) > 0

print(f"has_players_key: {has_players_key}")
print(f"players_raw_has_data: {players_raw_has_data}")
print(f"Evaluates to: {not has_players_key or not players_raw_has_data}")
print()

# Test the player extraction logic
print("PLAYER EXTRACTION:")
player_entries = []
for key, value in roster_wrapper.items():
    if key.isdigit() and isinstance(value, dict):
        player_entries.append(value)
        print(f"Found player under key '{key}': {value}")

print(f"Total players found: {len(player_entries)}")
print()

# Construct the players_raw dict
players_raw_new = {"player": player_entries}
print(f"players_raw_new: {players_raw_new}")
