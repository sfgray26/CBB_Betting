#!/usr/bin/env python3
"""
Comprehensive diagnostic for Discord vs UI discrepancy.
Checks all possible data sources.
"""

import os
import sys
import json
from datetime import datetime, timedelta, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_json_files():
    """Check all JSON files that might contain bet data."""
    print("=" * 70)
    print("📁 JSON FILES CHECK")
    print("=" * 70)
    
    files_to_check = [
        ("current_recommendations.json", "Current recommendations"),
        ("tmp_predictions.json", "Temp predictions"),
        ("tmp_bet_details.json", "Temp bet details"),
        ("full_today_data.json", "Full today data (Discord source)"),
        ("CUserssfgrareposFixedcbb-edgetmp_predictions.json", "Windows path temp"),
    ]
    
    for filename, description in files_to_check:
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Count bets
                bet_count = 0
                total_games = 0
                
                if isinstance(data, list):
                    bet_count = len(data)
                    # Check if items have verdict
                    bet_count = len([x for x in data if isinstance(x, dict) and x.get('verdict', '').startswith('Bet')])
                elif isinstance(data, dict):
                    total_games = data.get('total_games', 0)
                    bet_count = data.get('bets_recommended', 0)
                    if not bet_count and 'predictions' in data:
                        bet_count = len([p for p in data['predictions'] if p.get('verdict', '').startswith('Bet')])
                
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_hours = (datetime.now() - mtime).total_seconds() / 3600
                
                print(f"✅ {description}")
                print(f"   File: {filename}")
                print(f"   Bets: {bet_count} | Games: {total_games}")
                print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M')} ({age_hours:.1f}h ago)")
                print()
            except Exception as e:
                print(f"⚠️  {description}: Error reading - {e}")
                print()
        else:
            print(f"❌ {description}: File not found")
            print(f"   Expected: {filename}")
            print()


def analyze_discord_message_structure():
    """Analyze what structure Discord expects vs what it receives."""
    print("=" * 70)
    print("📱 DISCORD NOTIFIER ANALYSIS")
    print("=" * 70)
    
    print("Discord notifier expects:")
    print("  - bet_details: List of dicts with keys:")
    print("    - home_team, away_team, spread, bet_side")
    print("    - edge_conservative, recommended_units")
    print("    - verdict (should start with 'Bet')")
    print()
    
    # Check if we can find what was actually sent
    print("Checking tmp_bet_details.json content:")
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_bet_details.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"  Found {len(data)} entries")
            for i, entry in enumerate(data[:3]):  # Show first 3
                print(f"  [{i+1}] {entry.get('matchup', 'N/A')}")
                print(f"       Pick: {entry.get('pick', 'N/A')} | Edge: {entry.get('edge', 0):.2%}")
            if len(data) > 3:
                print(f"  ... and {len(data) - 3} more")
        print()


def check_timezones():
    """Check timezone-related issues."""
    print("=" * 70)
    print("🌍 TIMEZONE ANALYSIS")
    print("=" * 70)
    
    utc_now = datetime.utcnow()
    print(f"UTC now:        {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Common US timezones
    from datetime import timezone as tz
    
    # ET (UTC-5 or UTC-4 depending on DST)
    et_offset = timedelta(hours=-5)  # EST (ignoring DST for simplicity)
    et_now = utc_now.replace(tzinfo=tz.utc).astimezone(tz(timedelta(hours=-5)))
    print(f"ET (UTC-5):     {et_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # PT (UTC-8 or UTC-7 depending on DST)
    pt_now = utc_now.replace(tzinfo=tz.utc).astimezone(tz(timedelta(hours=-8)))
    print(f"PT (UTC-8):     {pt_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Your timezone (GMT+8)
    cn_now = utc_now.replace(tzinfo=tz.utc).astimezone(tz(timedelta(hours=8)))
    print(f"CN (UTC+8):     {cn_now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print()
    print("Today's date by timezone:")
    print(f"  UTC: {utc_now.date()}")
    print(f"  ET:  {et_now.date()}")
    print(f"  PT:  {pt_now.date()}")
    print(f"  CN:  {cn_now.date()}")
    
    # Check if dates differ
    dates = {utc_now.date(), et_now.date(), pt_now.date(), cn_now.date()}
    if len(dates) > 1:
        print()
        print("⚠️  DATES DIFFER ACROSS TIMEZONES!")
        print("   This could cause 'today' to mean different things")
    print()


def check_analysis_vs_predictions():
    """Check if analysis results differ from prediction query."""
    print("=" * 70)
    print("🔍 ANALYSIS vs PREDICTIONS CHECK")
    print("=" * 70)
    
    print("The discrepancy likely comes from:")
    print()
    print("1. ANALYSIS (what Discord sees):")
    print("   - Runs and finds games with edges")
    print("   - Sends to Discord immediately with bet_details")
    print("   - May include games that start 'today' in any timezone")
    print()
    print("2. UI /api/predictions/today (what you see):")
    print("   - Queries Prediction table for prediction_date = today")
    print("   - Filters Game.game_date > now_utc (upcoming only)")
    print("   - May exclude games that have already started")
    print()
    print("3. DIFFERENT 'TODAY' definitions:")
    print("   - Analysis might use a rolling 24h window")
    print("   - UI uses strict UTC date")
    print("   - Discord message might be from cached/stale data")
    print()


def suggest_fixes():
    """Suggest fixes based on diagnosis."""
    print("=" * 70)
    print("🔧 SUGGESTED FIXES")
    print("=" * 70)
    
    print("1. IMMEDIATE - Check what Discord actually received:")
    print("   - Look at the Discord message timestamp")
    print("   - Compare to current time")
    print("   - If message is old (>2h), it may be stale data")
    print()
    
    print("2. UI ENHANCEMENT - Already implemented:")
    print("   - Toggle to show 'all today' vs 'upcoming only'")
    print("   - Games marked with ⏱️ STARTED if they've begun")
    print()
    
    print("3. DATA CONSISTENCY:")
    print("   - Ensure analysis and UI use same 'today' definition")
    print("   - Consider using rolling 24h window everywhere")
    print("   - Or use strict UTC date everywhere")
    print()
    
    print("4. DEBUGGING:")
    print("   - Run: python scripts/diagnose_discord_ui_discrepancy.py")
    print("   - Check tmp_bet_details.json timestamp")
    print("   - Verify full_today_data.json exists and is current")
    print()


def main():
    print("=" * 70)
    print("🔍 COMPREHENSIVE DISCORD/UI DISCREPANCY DIAGNOSTIC")
    print("=" * 70)
    print(f"Current UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    check_timezones()
    check_json_files()
    analyze_discord_message_structure()
    check_analysis_vs_predictions()
    suggest_fixes()


if __name__ == "__main__":
    main()
