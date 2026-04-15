#!/usr/bin/env python3
"""
Comprehensive Database Health Audit - April 13, 2026
Assesses readiness for derived stats layer implementation
"""
import sys
import os
from sqlalchemy import create_engine, text
from datetime import datetime
import json

print("=" * 80)
print("DATABASE HEALTH AUDIT - Derived Stats Readiness Assessment")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
print("=" * 80)
print()

# Use Railway public proxy for external access
database_url = os.environ.get('DATABASE_PUBLIC_URL') or os.environ.get('DATABASE_URL')
if not database_url:
    print("ERROR: No database URL found")
    sys.exit(1)

print(f"Connecting using: {'DATABASE_PUBLIC_URL' if os.environ.get('DATABASE_PUBLIC_URL') else 'DATABASE_URL'}")
engine = create_engine(database_url, pool_pre_ping=True, connect_args={'connect_timeout': 30})
db = engine.connect()

try:
    audit = {
        'timestamp': datetime.now().isoformat(),
        'readiness_for_derived_stats': {},
        'table_health': {},
        'critical_nulls': {},
        'pipeline_freshness': {},
        'data_quality': {},
        'blockers': []
    }

    # ========================================================================
    # SECTION 1: Table Inventory & Row Counts
    # ========================================================================
    print("SECTION 1: TABLE INVENTORY")
    print("-" * 80)
    
    tables = [
        'mlb_games', 'mlb_player_stats', 'mlb_teams',
        'player_id_mapping', 'position_eligibility',
        'statcast_performances', 'probable_pitchers',
        'data_ingestion_logs', 'fantasy_leagues',
        'fantasy_teams', 'yahoo_rosters'
    ]
    
    for table in tables:
        try:
            count = db.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            audit['table_health'][table] = count
            status = 'OK' if count > 0 else 'EMPTY'
            if table in ['statcast_performances', 'mlb_player_stats', 'mlb_games'] and count == 0:
                status = 'CRITICAL'
            print(f"{status:12} {table:35} {count:>10,} rows")
        except Exception as e:
            audit['table_health'][table] = f"ERROR: {e}"
            print(f"ERROR        {table:35} {str(e)[:40]}")
    
    print()
    
    # ========================================================================
    # SECTION 2: Derived Stats Prerequisites
    # ========================================================================
    print("SECTION 2: DERIVED STATS READINESS")
    print("-" * 80)
    
    # 2.1 OPS backfill status
    print("\n2.1 OPS/WHIP Computation Status")
    ops_status = db.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE ops IS NULL) as null_ops,
            COUNT(*) FILTER (WHERE ops IS NULL AND obp IS NOT NULL AND slg IS NOT NULL) as null_ops_backfillable,
            COUNT(*) FILTER (WHERE ops IS NULL AND (obp IS NULL OR slg IS NULL)) as null_ops_unbackfillable,
            COUNT(*) FILTER (WHERE whip IS NULL) as null_whip,
            COUNT(*) FILTER (WHERE whip IS NULL AND p_bb IS NOT NULL AND p_hits IS NOT NULL AND ip IS NOT NULL AND ip != '' AND ip != '0.0') as null_whip_backfillable,
            COUNT(*) FILTER (WHERE whip IS NULL AND (p_bb IS NULL OR p_hits IS NULL OR ip IS NULL OR ip = '' OR ip = '0.0')) as null_whip_unbackfillable
        FROM mlb_player_stats
    """)).fetchone()
    
    audit['readiness_for_derived_stats']['ops_whip'] = {
        'total_rows': ops_status.total,
        'null_ops': ops_status.null_ops,
        'null_ops_pct': round(ops_status.null_ops / ops_status.total * 100, 1) if ops_status.total else 0,
        'backfillable_ops': ops_status.null_ops_backfillable,
        'unbackfillable_ops': ops_status.null_ops_unbackfillable,
        'null_whip': ops_status.null_whip,
        'null_whip_pct': round(ops_status.null_whip / ops_status.total * 100, 1) if ops_status.total else 0,
        'backfillable_whip': ops_status.null_whip_backfillable,
        'unbackfillable_whip': ops_status.null_whip_unbackfillable,
    }
    
    print(f"  Total rows:        {ops_status.total:,}")
    print(f"  NULL OPS:          {ops_status.null_ops:,} ({audit['readiness_for_derived_stats']['ops_whip']['null_ops_pct']}%)")
    print(f"    └─ Backfillable: {ops_status.null_ops_backfillable:,}")
    print(f"    └─ Unbackfillable: {ops_status.null_ops_unbackfillable:,}")
    print(f"  NULL WHIP:         {ops_status.null_whip:,} ({audit['readiness_for_derived_stats']['ops_whip']['null_whip_pct']}%)")
    print(f"    └─ Backfillable: {ops_status.null_whip_backfillable:,}")
    print(f"    └─ Unbackfillable: {ops_status.null_whip_unbackfillable:,}")
    
    if ops_status.null_ops_backfillable > 0:
        audit['blockers'].append(f"{ops_status.null_ops_backfillable} rows need OPS backfill")
    if ops_status.null_whip_backfillable > 0:
        audit['blockers'].append(f"{ops_status.null_whip_backfillable} rows need WHIP backfill")
    
    # 2.2 Caught stealing / NSB
    print("\n2.2 Net Stolen Bases (NSB = SB - CS) Status")
    cs_status = db.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE caught_stealing IS NULL) as null_cs,
            COUNT(*) FILTER (WHERE caught_stealing IS NOT NULL) as has_cs,
            COUNT(*) FILTER (WHERE stolen_bases IS NULL) as null_sb
        FROM mlb_player_stats
    """)).fetchone()
    
    audit['readiness_for_derived_stats']['nsb'] = {
        'total_rows': cs_status.total,
        'null_caught_stealing': cs_status.null_cs,
        'null_caught_stealing_pct': round(cs_status.null_cs / cs_status.total * 100, 1) if cs_status.total else 0,
        'has_caught_stealing': cs_status.has_cs,
        'null_stolen_bases': cs_status.null_sb
    }
    
    print(f"  NULL caught_stealing: {cs_status.null_cs:,} ({audit['readiness_for_derived_stats']['nsb']['null_caught_stealing_pct']}%)")
    print(f"  Has caught_stealing:  {cs_status.has_cs:,}")
    print(f"  NULL stolen_bases:    {cs_status.null_sb:,}")
    
    if cs_status.null_cs > 0 and cs_status.null_cs == cs_status.total:
        audit['blockers'].append("caught_stealing is 100% NULL - NSB cannot be computed")
    
    # 2.3 Player identity linkage
    print("\n2.3 Player Identity Linkage Status")
    identity_status = db.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM player_id_mapping) as total_mapping,
            (SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NOT NULL) as has_yahoo_key,
            (SELECT COUNT(*) FROM player_id_mapping WHERE mlbam_id IS NOT NULL) as has_mlbam_id,
            (SELECT COUNT(*) FROM position_eligibility) as total_positions,
            (SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NOT NULL) as linked_positions
    """)).fetchone()
    
    audit['readiness_for_derived_stats']['player_identity'] = {
        'total_mapping': identity_status.total_mapping,
        'yahoo_key_coverage': round(identity_status.has_yahoo_key / identity_status.total_mapping * 100, 1) if identity_status.total_mapping else 0,
        'mlbam_id_coverage': round(identity_status.has_mlbam_id / identity_status.total_mapping * 100, 1) if identity_status.total_mapping else 0,
        'position_linkage': round(identity_status.linked_positions / identity_status.total_positions * 100, 1) if identity_status.total_positions else 0,
        'orphan_positions': identity_status.total_positions - identity_status.linked_positions
    }
    
    print(f"  player_id_mapping rows:     {identity_status.total_mapping:,}")
    print(f"  With yahoo_key:             {identity_status.has_yahoo_key:,} ({audit['readiness_for_derived_stats']['player_identity']['yahoo_key_coverage']}%)")
    print(f"  With mlbam_id:              {identity_status.has_mlbam_id:,} ({audit['readiness_for_derived_stats']['player_identity']['mlbam_id_coverage']}%)")
    print(f"  position_eligibility rows:  {identity_status.total_positions:,}")
    print(f"  Linked to bdl_player_id:    {identity_status.linked_positions:,} ({audit['readiness_for_derived_stats']['player_identity']['position_linkage']}%)")
    print(f"  Orphaned positions:         {identity_status.total_positions - identity_status.linked_positions:,}")
    
    # 2.4 Statcast status
    print("\n2.4 Statcast Data Status")
    statcast_status = db.execute(text("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT player_id) as unique_players,
            MIN(game_date) as earliest,
            MAX(game_date) as latest,
            COUNT(*) FILTER (WHERE launch_speed IS NULL) as null_launch_speed,
            COUNT(*) FILTER (WHERE xwoba IS NULL) as null_xwoba,
            COUNT(*) FILTER (WHERE barrel IS NULL) as null_barrel
        FROM statcast_performances
    """)).fetchone()
    
    if statcast_status.total_rows == 0:
        audit['readiness_for_derived_stats']['statcast'] = {
            'total_rows': 0,
            'status': 'EMPTY'
        }
        print(f"  CRITICAL: Table is EMPTY")
        audit['blockers'].append("statcast_performances is empty - advanced metrics unavailable")
    else:
        audit['readiness_for_derived_stats']['statcast'] = {
            'total_rows': statcast_status.total_rows,
            'unique_players': statcast_status.unique_players,
            'earliest': str(statcast_status.earliest) if statcast_status.earliest else None,
            'latest': str(statcast_status.latest) if statcast_status.latest else None,
            'null_launch_speed': statcast_status.null_launch_speed,
            'null_xwoba': statcast_status.null_xwoba,
            'null_barrel': statcast_status.null_barrel
        }
        print(f"  ✅ Total rows:         {statcast_status.total_rows:,}")
        print(f"  Unique players:       {statcast_status.unique_players:,}")
        print(f"  Date range:           {statcast_status.earliest} to {statcast_status.latest}")
        print(f"  NULL launch_speed:    {statcast_status.null_launch_speed:,}")
        print(f"  NULL xwOBA:           {statcast_status.null_xwoba:,}")
        print(f"  NULL barrel:          {statcast_status.null_barrel:,}")
    
    # ========================================================================
    # SECTION 3: Pipeline Freshness
    # ========================================================================
    print("\n\nSECTION 3: PIPELINE FRESHNESS")
    print("-" * 80)
    
    freshness = db.execute(text("""
        SELECT
            (SELECT MAX(game_date) FROM mlb_games) as latest_game,
            (SELECT MAX(created_at) FROM mlb_player_stats) as latest_stats,
            (SELECT MAX(updated_at) FROM yahoo_rosters) as latest_roster,
            (SELECT COUNT(*) FROM mlb_games WHERE game_date >= CURRENT_DATE - INTERVAL '7 days') as games_7d
    """)).fetchone()
    
    audit['pipeline_freshness'] = {
        'latest_game': str(freshness.latest_game) if freshness.latest_game else None,
        'latest_stats': str(freshness.latest_stats) if freshness.latest_stats else None,
        'latest_roster': str(freshness.latest_roster) if freshness.latest_roster else None,
        'games_last_7d': freshness.games_7d
    }
    
    print(f"  Latest MLB game:      {freshness.latest_game}")
    print(f"  Latest player stats:  {freshness.latest_stats}")
    print(f"  Latest roster sync:   {freshness.latest_roster}")
    print(f"  Games in last 7 days: {freshness.games_7d:,}")
    
    if freshness.games_7d == 0:
        audit['blockers'].append("No games ingested in last 7 days - pipeline may be stalled")
    
    # ========================================================================
    # SECTION 4: Data Quality Checks
    # ========================================================================
    print("\n\nSECTION 4: DATA QUALITY CHECKS")
    print("-" * 80)
    
    # 4.1 Impossible values
    dq = db.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE era IS NOT NULL AND (era < 0 OR era > 100)) as bad_era,
            COUNT(*) FILTER (WHERE avg IS NOT NULL AND (avg < 0 OR avg > 1)) as bad_avg,
            COUNT(*) FILTER (WHERE ops IS NOT NULL AND ops < 0) as bad_ops,
            COUNT(*) FILTER (WHERE whip IS NOT NULL AND whip < 0) as bad_whip
        FROM mlb_player_stats
    """)).fetchone()
    
    audit['data_quality'] = {
        'bad_era': dq.bad_era,
        'bad_avg': dq.bad_avg,
        'bad_ops': dq.bad_ops,
        'bad_whip': dq.bad_whip
    }
    
    print(f"  Bad ERA values:       {dq.bad_era:,}")
    print(f"  Bad AVG values:       {dq.bad_avg:,}")
    print(f"  Bad OPS values:       {dq.bad_ops:,}")
    print(f"  Bad WHIP values:      {dq.bad_whip:,}")
    
    if dq.bad_era > 0 or dq.bad_avg > 0:
        audit['blockers'].append(f"Data quality issues: {dq.bad_era} bad ERA, {dq.bad_avg} bad AVG")
    
    # ========================================================================
    # SECTION 5: Derived Stats Readiness Verdict
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("SECTION 5: DERIVED STATS READINESS VERDICT")
    print("=" * 80)
    
    # Score each component
    scores = {}
    
    # OPS/WHIP score
    if ops_status.null_ops_backfillable == 0 and ops_status.null_ops_unbackfillable < 500:
        scores['ops_whip'] = 95
        print("\n✅ OPS/WHIP:        READY (95/100)")
    elif ops_status.null_ops_backfillable == 0:
        scores['ops_whip'] = 80
        print("\n⚠️  OPS/WHIP:        PARTIAL (80/100) - some unbackfillable NULLs")
    else:
        scores['ops_whip'] = max(0, 100 - int(ops_status.null_ops_backfillable / ops_status.total * 100))
        print(f"\n🔴 OPS/WHIP:        NOT READY ({scores['ops_whip']}/100) - {ops_status.null_ops_backfillable:,} need backfill")
    
    # NSB score
    if cs_status.has_cs > 0:
        scores['nsb'] = 95
        print("✅ NSB (SB-CS):      READY (95/100)")
    else:
        scores['nsb'] = 0
        print("🔴 NSB (SB-CS):      NOT READY (0/100) - caught_stealing unavailable")
    
    # Identity linkage score
    pos_link_pct = identity_status.linked_positions / identity_status.total_positions if identity_status.total_positions else 0
    scores['identity'] = int(pos_link_pct * 100)
    if pos_link_pct >= 0.95:
        print(f"✅ Identity Linkage: READY ({scores['identity']}/100)")
    elif pos_link_pct >= 0.80:
        print(f"⚠️  Identity Linkage: PARTIAL ({scores['identity']}/100)")
    else:
        print(f"🔴 Identity Linkage: NOT READY ({scores['identity']}/100)")
    
    # Statcast score
    if statcast_status.total_rows > 10000:
        scores['statcast'] = 95
        print(f"✅ Statcast:         READY ({scores['statcast']}/100) - {statcast_status.total_rows:,} rows")
    elif statcast_status.total_rows > 0:
        scores['statcast'] = 50
        print(f"⚠️  Statcast:         PARTIAL ({scores['statcast']}/100) - {statcast_status.total_rows:,} rows")
    else:
        scores['statcast'] = 0
        print("🔴 Statcast:         NOT READY (0/100) - table empty")
    
    # Pipeline freshness score
    if freshness.games_7d > 10:
        scores['freshness'] = 95
        print(f"✅ Pipeline Freshness: READY ({scores['freshness']}/100)")
    elif freshness.games_7d > 0:
        scores['freshness'] = 70
        print(f"⚠️  Pipeline Freshness: STALE ({scores['freshness']}/100)")
    else:
        scores['freshness'] = 0
        print("🔴 Pipeline Freshness: NOT READY (0/100)")
    
    # Overall score
    overall = sum(scores.values()) / len(scores)
    audit['overall_readiness_score'] = round(overall, 1)
    
    print(f"\n{'=' * 40}")
    print(f"OVERALL READINESS SCORE: {overall:.1f}/100")
    print(f"{'=' * 40}")
    
    if overall >= 85:
        verdict = "READY for derived stats layer"
        print(f"\nVERDICT: {verdict}")
    elif overall >= 65:
        verdict = "MARGINAL - address blockers before building derived stats"
        print(f"\nVERDICT: {verdict}")
    else:
        verdict = "NOT READY - fix pipeline issues first"
        print(f"\nVERDICT: {verdict}")
    
    audit['verdict'] = verdict
    
    # Blockers
    if audit['blockers']:
        print(f"\nACTIVE BLOCKERS ({len(audit['blockers'])}):")
        for i, blocker in enumerate(audit['blockers'], 1):
            print(f"  {i}. {blocker}")
    else:
        print("\nNo active blockers found.")
    
    # Save report
    output_file = f'reports/{datetime.now().strftime("%Y-%m-%d")}-db-health-readiness-audit.json'
    with open(output_file, 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    
    print(f"\nFull audit saved to: {output_file}")
    print("\nAudit complete!")

finally:
    db.close()
    engine.dispose()
