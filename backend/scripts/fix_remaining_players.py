"""
Fix remaining active players: Jordan Walker, Cristopher Sánchez, verify Edwin Díaz IL.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, PlayerIDMapping, IngestedInjury, PlayerScore


def check_jordan_walker(db):
    """Check Jordan Walker's current mapping status and fix."""
    print("\n" + "="*60)
    print("JORDAN WALKER")
    print("="*60)

    # Find all Jordan Walker mappings
    mappings = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.full_name.ilike("%jordan walker%")
    ).all()

    print(f"Found {len(mappings)} mappings:")
    for m in mappings:
        print(f"  id={m.id}, yahoo_key={m.yahoo_key}, bdl_id={m.bdl_id}, mlbam_id={m.mlbam_id}")

    # We know from earlier research that bdl_id=539 has 85 games of stats
    # Check if that mapping exists
    bdl_539 = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.bdl_id == 539
    ).first()

    if bdl_539:
        print(f"\nFound mapping with bdl_id=539:")
        print(f"  id={bdl_539.id}, full_name={bdl_539.full_name}")
        print(f"  yahoo_key={bdl_539.yahoo_key}")

        if not bdl_539.yahoo_key:
            print("\n  *** ACTION NEEDED: Add yahoo_key to this row ***")
            # Need to find Jordan Walker's yahoo_key
            # From earlier: we need to discover it from Yahoo API
            print("  Need to discover yahoo_key from Yahoo roster API")
            return "NEEDS_YAHOO_KEY"
        else:
            print("\n  Already has yahoo_key - should be working")
            return "OK"
    else:
        print("\n  *** No mapping with bdl_id=539 found - need to create ***")
        return "NEEDS_MAPPING_CREATION"


def check_cristopher_sanchez(db):
    """Check Cristopher Sánchez mapping status."""
    print("\n" + "="*60)
    print("CRISTOPHER SÁNCHEZ")
    print("="*60)

    # Find all Sánchez mappings (could be Sanchez without accent)
    mappings = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.full_name.ilike("%sanchez%")
    ).all()

    # Filter for Cristopher specifically
    cristopher_mappings = [m for m in mappings if m.full_name and "cristopher" in m.full_name.lower()]

    print(f"Found {len(cristopher_mappings)} Cristopher Sánchez mappings:")
    for m in cristopher_mappings:
        print(f"  id={m.id}, yahoo_key={m.yahoo_key}, bdl_id={m.bdl_id}, mlbam_id={m.mlbam_id}")

    if cristopher_mappings:
        # Check if any have yahoo_key
        with_key = [m for m in cristopher_mappings if m.yahoo_key]
        if with_key:
            print("\n  Has yahoo_key mapping - should be working")
            return "OK"
        else:
            print("\n  *** ACTION NEEDED: Add yahoo_key to existing row ***")
            return "NEEDS_YAHOO_KEY"
    else:
        print("\n  *** No Cristopher Sánchez mapping found - need to create ***")
        return "NEEDS_FULL_MAPPING"


def check_edwin_diaz(db):
    """Check Edwin Díaz - verify IL exclusion works."""
    print("\n" + "="*60)
    print("EDWIN DÍAZ")
    print("="*60)

    # Find Díaz mappings
    mappings = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.full_name.ilike("%diaz%")
    ).all()

    # Filter for Edwin
    edwin_mappings = [m for m in mappings if m.full_name and "edwin" in m.full_name.lower()]

    print(f"Found {len(edwin_mappings)} Edwin Díaz mappings:")
    for m in edwin_mappings:
        print(f"  id={m.id}, yahoo_key={m.yahoo_key}, bdl_id={m.bdl_id}, mlbam_id={m.mlbam_id}")

    # Check injury overlays
    print("\nChecking injury overlays for Edwin Díaz:")
    overlays = db.query(IngestedInjury).filter(
        IngestedInjury.player_name.ilike("%edwin%diaz%")
    ).all()

    print(f"Found {len(overlays)} injury overlays:")
    for o in overlays:
        print(f"  injury_status={o.injury_status}, note={o.short_comment}")

    if overlays:
        # Check if any show IL status
        il_overlays = [o for o in overlays if o.injury_status and "il" in o.injury_status.lower()]
        if il_overlays:
            print("\n  *** Edwin Díaz has IL status - should be excluded from optimizer ***")
            return "IL_EXCLUDED"
        else:
            print("\n  Has injury overlay but not IL - may still be active")
    else:
        print("\n  No injury overlays found")

    # If no IL status and no mapping, he needs mapping
    if not edwin_mappings:
        print("\n  *** No mapping found, but appears to be IL (surgical recovery) ***")
        print("  *** Verify: Should he be excluded via IL or added as active player? ***")
        return "NEEDS_VERIFICATION"

    return "OK"


def discover_yahoo_keys_from_yahoo(db):
    """
    Discover yahoo_keys from Yahoo roster API.

    This requires calling the Yahoo Fantasy API to get the current roster
    and extracting player keys for Jordan Walker and Cristopher Sánchez.
    """
    print("\n" + "="*60)
    print("DISCOVERING YAHOO KEYS FROM YAHOO API")
    print("="*60)

    try:
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client

        client = get_yahoo_client()
        roster = client.get_roster()

        # Look for our target players
        target_names = ["Jordan Walker", "Cristopher Sánchez", "Edwin Díaz"]
        found = {}

        for player in roster:
            name = player.get("name", "")
            player_key = player.get("player_key", "")

            for target in target_names:
                if target.lower() in name.lower():
                    found[target] = {
                        "name": name,
                        "player_key": player_key,
                        "status": player.get("status"),
                    }

        print("\nFound players on Yahoo roster:")
        for target, info in found.items():
            print(f"  {target}:")
            print(f"    Yahoo Key: {info['player_key']}")
            print(f"    Status: {info['status']}")

        return found

    except Exception as e:
        print(f"ERROR: Could not fetch Yahoo roster: {e}")
        return {}


def main():
    db = SessionLocal()

    try:
        print("="*60)
        print("FIX REMAINING ACTIVE PLAYERS")
        print("="*60)

        # Check current status
        walker_status = check_jordan_walker(db)
        sanchez_status = check_cristopher_sanchez(db)
        diaz_status = check_edwin_diaz(db)

        # Try to discover yahoo keys
        yahoo_players = discover_yahoo_keys_from_yahoo(db)

        # Apply fixes
        print("\n" + "="*60)
        print("APPLYING FIXES")
        print("="*60)

        # Jordan Walker
        if "Jordan Walker" in yahoo_players and walker_status != "OK":
            yahoo_key = yahoo_players["Jordan Walker"]["player_key"]
            print(f"\nFixing Jordan Walker: Adding yahoo_key={yahoo_key} to bdl_id=539")

            # Find the row with bdl_id=539
            row = db.query(PlayerIDMapping).filter(
                PlayerIDMapping.bdl_id == 539
            ).first()

            if row:
                row.yahoo_key = yahoo_key
                if ".p." in yahoo_key:
                    row.yahoo_id = yahoo_key.split(".p.", 1)[-1]
                db.commit()
                print(f"  UPDATED: Added yahoo_key={yahoo_key} to row id={row.id}")
            else:
                print(f"  ERROR: No row with bdl_id=539 found")

        # Cristopher Sánchez
        if "Cristopher Sánchez" in yahoo_players and sanchez_status != "OK":
            yahoo_key = yahoo_players["Cristopher Sánchez"]["player_key"]
            print(f"\nFixing Cristopher Sánchez: Adding yahoo_key={yahoo_key}")

            # Find if a mapping exists without yahoo_key
            row = db.query(PlayerIDMapping).filter(
                PlayerIDMapping.full_name.ilike("%cristopher%sanchez%"),
                PlayerIDMapping.yahoo_key.is_(None)
            ).first()

            if row:
                row.yahoo_key = yahoo_key
                if ".p." in yahoo_key:
                    row.yahoo_id = yahoo_key.split(".p.", 1)[-1]
                db.commit()
                print(f"  UPDATED: Added yahoo_key={yahoo_key} to existing row id={row.id}")
            else:
                print(f"  ERROR: No existing row found - need to create new mapping (requires bdl_id)")

        # Edwin Díaz
        if "Edwin Díaz" in yahoo_players:
            yahoo_key = yahoo_players["Edwin Díaz"]["player_key"]
            status = yahoo_players["Edwin Díaz"]["status"]
            print(f"\nEdwin Díaz found on Yahoo:")
            print(f"  Yahoo Key: {yahoo_key}")
            print(f"  Status: {status}")

            if status and "il" in status.lower():
                print(f"  *** CONFIRMED: Edwin Díaz is IL - should be excluded from optimizer ***")
            else:
                print(f"  *** WARNING: Edwin Díaz status is '{status}' - not IL! ***")
                print(f"  *** May need mapping if he's active ***")

    finally:
        db.close()


if __name__ == "__main__":
    main()
