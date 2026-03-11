#!/usr/bin/env python3
"""
Analyze betting results from CSV export with critical lens.
"""

import csv
from collections import defaultdict

def analyze_results(csv_path):
    """Analyze betting results and identify patterns."""
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print("=" * 80)
    print("🔍 BETTING RESULTS ANALYSIS — CRITICAL REVIEW")
    print("=" * 80)
    print(f"\nTotal Bets Analyzed: {len(rows)}")
    
    # Separate real vs paper trades
    real_bets = [r for r in rows if r.get('Paper?', '').lower() == 'false']
    paper_bets = [r for r in rows if r.get('Paper?', '').lower() == 'true']
    
    print(f"Real Bets: {len(real_bets)}")
    print(f"Paper Bets: {len(paper_bets)}")
    
    # Focus on real bets with actual P&L
    real_with_pl = [r for r in real_bets if float(r.get('P&L ($)', 0) or 0) != 0]
    
    if not real_with_pl:
        print("\n⚠️  No real bets with actual P&L found in this export")
        return
    
    print(f"\nReal Bets with P&L: {len(real_with_pl)}")
    
    # Calculate metrics
    wins = len([r for r in real_with_pl if r.get('Result', '').lower() == 'win'])
    losses = len([r for r in real_with_pl if r.get('Result', '').lower() == 'loss'])
    pushes = len([r for r in real_with_pl if r.get('Result', '').lower() == 'push'])
    
    total_pl = sum(float(r.get('P&L ($)', 0) or 0) for r in real_with_pl)
    total_units = sum(float(r.get('P&L (u)', 0) or 0) for r in real_with_pl)
    total_risked = sum(float(r.get('Risk ($)', 0) or 0) for r in real_with_pl)
    
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    roi = total_pl / total_risked if total_risked > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 OVERALL METRICS (Real Bets Only)")
    print("=" * 80)
    print(f"Wins: {wins} | Losses: {losses} | Pushes: {pushes}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total P&L: ${total_pl:+.2f}")
    print(f"Total Units: {total_units:+.2f}u")
    print(f"ROI: {roi:.1%}")
    print(f"Avg CLV: {sum(float(r.get('CLV pts', 0) or 0) for r in real_with_pl) / len(real_with_pl):+.2f} pts")
    
    # Group by matchup
    print("\n" + "=" * 80)
    print("🎯 BY MATCHUP")
    print("=" * 80)
    
    by_matchup = defaultdict(list)
    for r in real_with_pl:
        matchup = r.get('Matchup', 'Unknown')
        by_matchup[matchup].append(r)
    
    for matchup, bets in sorted(by_matchup.items(), key=lambda x: len(x[1]), reverse=True):
        wins_m = len([b for b in bets if b.get('Result', '').lower() == 'win'])
        losses_m = len([b for b in bets if b.get('Result', '').lower() == 'loss'])
        pl_m = sum(float(b.get('P&L ($)', 0) or 0) for b in bets)
        print(f"\n{matchup}")
        print(f"  Bets: {len(bets)} | W/L: {wins_m}/{losses_m} | P&L: ${pl_m:+.2f}")
        for b in bets:
            clv = float(b.get('CLV pts', 0) or 0)
            print(f"    {b.get('Pick', 'N/A')} | {b.get('Result', 'N/A')} | CLV: {clv:+.1f}pts")
    
    # CLV Analysis
    print("\n" + "=" * 80)
    print("📈 CLV ANALYSIS")
    print("=" * 80)
    
    positive_clv = [r for r in real_with_pl if float(r.get('CLV pts', 0) or 0) > 0]
    negative_clv = [r for r in real_with_pl if float(r.get('CLV pts', 0) or 0) < 0]
    
    print(f"Positive CLV bets: {len(positive_clv)}")
    if positive_clv:
        wins_pos = len([r for r in positive_clv if r.get('Result', '').lower() == 'win'])
        print(f"  Win rate: {wins_pos / len(positive_clv):.1%}")
        print(f"  Avg CLV: {sum(float(r.get('CLV pts', 0) or 0) for r in positive_clv) / len(positive_clv):+.2f} pts")
    
    print(f"\nNegative CLV bets: {len(negative_clv)}")
    if negative_clv:
        wins_neg = len([r for r in negative_clv if r.get('Result', '').lower() == 'win'])
        print(f"  Win rate: {wins_neg / len(negative_clv):.1%}")
        print(f"  Avg CLV: {sum(float(r.get('CLV pts', 0) or 0) for r in negative_clv) / len(negative_clv):+.2f} pts")
    
    # Unit size analysis
    print("\n" + "=" * 80)
    print("💰 UNIT SIZE ANALYSIS")
    print("=" * 80)
    
    by_units = defaultdict(list)
    for r in real_with_pl:
        units = float(r.get('Units', 0) or 0)
        by_units[units].append(r)
    
    for units in sorted(by_units.keys(), reverse=True):
        bets = by_units[units]
        wins_u = len([b for b in bets if b.get('Result', '').lower() == 'win'])
        pl_u = sum(float(b.get('P&L ($)', 0) or 0) for b in bets)
        print(f"{units}u bets: {len(bets)} | W: {wins_u} | P&L: ${pl_u:+.2f}")
    
    # Red flags
    print("\n" + "=" * 80)
    print("🚨 RED FLAGS")
    print("=" * 80)
    
    # Same game multiple bets
    print("\n1. Multiple bets on same game:")
    for matchup, bets in by_matchup.items():
        if len(bets) > 1:
            print(f"   {matchup}: {len(bets)} bets")
    
    # Negative CLV wins (lucky)
    print("\n2. Wins with negative CLV (lucky wins, not skill):")
    for r in positive_clv:
        if r.get('Result', '').lower() == 'loss':
            print(f"   ❌ {r.get('Matchup', 'N/A')} | Lost despite +{float(r.get('CLV pts', 0)):.1f} CLV")
    
    # High unit losses
    print("\n3. High unit bets that lost:")
    high_unit_losses = [r for r in real_with_pl 
                        if float(r.get('Units', 0) or 0) >= 1.0 
                        and r.get('Result', '').lower() == 'loss']
    for r in sorted(high_unit_losses, key=lambda x: float(x.get('Units', 0) or 0), reverse=True):
        print(f"   ❌ {r.get('Matchup', 'N/A')} | {float(r.get('Units', 0))}u | ${float(r.get('P&L ($)', 0)):.2f}")


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/root/openclaw/kimi/downloads/19cdaa0a-eea2-8053-8000-0000fe22cac1_2026-03-11T02-00_export.csv"
    analyze_results(csv_path)
