"""
Test script to verify IL exclusion and position eligibility fixes.

Validates:
1. Garrett Crochet (IL/Shoulder) is NOT in active slots
2. All UTIL players have hitting positions (C, 1B, 2B, 3B, SS, OF, DH)
3. No IL players in active lineup
"""

import requests
import json
import sys

# Production API endpoint
API_URL = "https://fantasy-app-production-5079.up.railway.app/api/fantasy/roster/optimize"
API_KEY = "your-api-key-here"  # Replace with actual API key if needed


def test_optimizer():
    """Test the optimizer endpoint and validate the results."""

    print("=" * 70)
    print("IL EXCLUSION AND POSITION ELIGIBILITY TEST")
    print("=" * 70)

    # Call optimizer
    print("\n1. Calling optimizer endpoint...")
    try:
        headers = {}
        # if API_KEY != "your-api-key-here":
        #     headers["X-API-Key"] = API_KEY

        response = requests.post(API_URL, json={}, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"   Status: {response.status_code}")
        print(f"   Success: {data.get('success')}")
        print(f"   Message: {data.get('message')}")

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    # Check for IL exclusion in message
    print("\n2. Checking IL exclusion message...")
    message = data.get('message', '')
    if 'IL' in message and 'excluded' in message.lower():
        print(f"   [OK] IL exclusion mentioned in message")
        print(f"   Message: {message}")
    else:
        print(f"   Note: IL exclusion not explicitly mentioned")

    # Analyze starters
    print("\n3. Analyzing active lineup...")
    starters = data.get('starters', [])

    # Check for Crochet
    crochet_in_starters = any(
        'Crochet' in starter.get('name', '')
        for starter in starters
    )

    if crochet_in_starters:
        print(f"   [X] FAIL: Garrett Crochet found in active lineup!")
        for starter in starters:
            if 'Crochet' in starter.get('name', ''):
                print(f"      - {starter.get('name')} in {starter.get('slot')} slot")
        return False
    else:
        print(f"   [OK] PASS: Garrett Crochet NOT in active lineup")

    # Check UTIL position eligibility
    print("\n4. Checking UTIL position eligibility...")
    util_players = [
        s for s in starters
        if s.get('slot') in ['UTIL', 'Util', 'utility']
    ]

    hitting_positions = {'C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH'}
    pitcher_positions = {'SP', 'RP', 'P'}

    all_valid = True
    for player in util_players:
        name = player.get('name', 'Unknown')
        positions = player.get('eligible_positions', [])

        # Check if player has pitcher positions
        has_pitcher_pos = any(pos.upper() in pitcher_positions for pos in positions if pos)

        # Check if player has hitting positions
        has_hitting_pos = any(pos.upper() in hitting_positions for pos in positions if pos)

        if has_pitcher_pos and not has_hitting_pos:
            print(f"   [X] FAIL: {name} in UTIL with only pitcher positions: {positions}")
            all_valid = False
        elif not has_hitting_pos and not positions:
            print(f"   [X] FAIL: {name} in UTIL with no positions")
            all_valid = False
        else:
            print(f"   [OK] {name} in UTIL - positions: {positions}")

    if all_valid:
        print(f"   [OK] PASS: All UTIL players have hitting positions")

    # Check for any IL players in active slots
    print("\n5. Checking for IL players in active slots...")
    il_keywords = ['IL', 'DL', 'OUT', 'DTD', 'Shoulder', 'Elbow', 'Knee', 'Arm',
                   'Finger', 'Wrist', 'Back', 'Hip', 'Hamstring', 'Quad', 'Ankle']

    il_in_active = []
    for starter in starters:
        name = starter.get('name', '')
        status = starter.get('status', '')
        injury_note = starter.get('injury_note', '')

        for keyword in il_keywords:
            if keyword.lower() in status.lower() or keyword.lower() in injury_note.lower():
                il_in_active.append({
                    'name': name,
                    'slot': starter.get('slot'),
                    'status': status,
                    'note': injury_note
                })
                break

    if il_in_active:
        print(f"   [X] FAIL: Found IL players in active slots:")
        for player in il_in_active:
            print(f"      - {player['name']} ({player['slot']}): {player['status']} - {player['note']}")
        return False
    else:
        print(f"   [OK] PASS: No IL players in active slots")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total active players: {len(starters)}")
    print(f"UTIL players: {len(util_players)}")
    print(f"Crochet in active: {crochet_in_starters}")
    print(f"IL players in active: {len(il_in_active)}")
    print(f"All UTIL valid: {all_valid}")

    if not crochet_in_starters and all_valid and not il_in_active:
        print("\n[OK][OK][OK] ALL TESTS PASSED [OK][OK][OK]")
        return True
    else:
        print("\n[X][X][X] SOME TESTS FAILED [X][X][X]")
        return False


if __name__ == "__main__":
    success = test_optimizer()
    sys.exit(0 if success else 1)
