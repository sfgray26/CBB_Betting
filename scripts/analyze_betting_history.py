#!/usr/bin/env python3
"""Analyze CBB betting history to identify performance issues."""

import csv
from collections import defaultdict
from datetime import datetime
import json

def analyze_betting():
    # Read betting history
    wagers = []
    winnings = []
    deposits = []
    
    with open('draftkings_history.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Category'] == 'Wagers and Entry Fees and Purchases':
                wagers.append(row)
            elif row['Category'] == 'Winnings':
                winnings.append(row)
            elif row['Category'] == 'Deposits':
                deposits.append(row)
    
    # Calculate totals
    total_wagered = sum(float(w['AmountRaw']) for w in wagers)
    total_won = sum(float(w['AmountRaw']) for w in winnings)
    total_deposited = sum(float(d['AmountRaw']) for d in deposits)
    net_pnl = total_wagered + total_won  # wagers are negative
    
    print("=" * 60)
    print("CBB BETTING PERFORMANCE ANALYSIS")
    print("=" * 60)
    print()
    print(f"Total Wagers: {len(wagers)}")
    print(f"Total Wins: {len(winnings)}")
    print(f"Total Deposits: ${total_deposited:.2f}")
    print()
    print(f"Total Wagered: ${abs(total_wagered):.2f}")
    print(f"Total Won: ${total_won:.2f}")
    print(f"Net P&L: ${net_pnl:.2f}")
    print(f"ROI: {(net_pnl/abs(total_wagered)*100):.1f}%")
    print()
    
    # Win rate
    win_rate = len(winnings) / len(wagers) * 100 if wagers else 0
    print(f"Win Rate: {win_rate:.1f}%")
    
    # Average bet size
    avg_bet = abs(total_wagered) / len(wagers) if wagers else 0
    avg_win = total_won / len(winnings) if winnings else 0
    print(f"Average Bet: ${avg_bet:.2f}")
    print(f"Average Win: ${avg_win:.2f}")
    print()
    
    # Group by date
    daily_pnl = defaultdict(lambda: {'wagered': 0, 'won': 0, 'net': 0})
    for w in wagers:
        try:
            date = w.get('Date', '').split()[0] if w.get('Date') else 'Unknown'
            if date != 'Unknown':
                daily_pnl[date]['wagered'] += abs(float(w['AmountRaw']))
                daily_pnl[date]['net'] += float(w['AmountRaw'])
        except (KeyError, IndexError, ValueError):
            continue
    
    for w in winnings:
        try:
            date = w.get('Date', '').split()[0] if w.get('Date') else 'Unknown'
            if date != 'Unknown':
                daily_pnl[date]['won'] += float(w['AmountRaw'])
                daily_pnl[date]['net'] += float(w['AmountRaw'])
        except (KeyError, IndexError, ValueError):
            continue
    
    # Find best and worst days
    sorted_days = sorted(daily_pnl.items(), key=lambda x: x[1]['net'], reverse=True)
    
    print("=" * 60)
    print("TOP 5 BEST DAYS")
    print("=" * 60)
    for date, data in sorted_days[:5]:
        if data['net'] > 0:
            print(f"{date}: +${data['net']:.2f} (wagered ${data['wagered']:.2f}, won ${data['won']:.2f})")
    
    print()
    print("=" * 60)
    print("TOP 5 WORST DAYS")
    print("=" * 60)
    for date, data in sorted_days[-5:]:
        if data['net'] < 0:
            print(f"{date}: ${data['net']:.2f} (wagered ${data['wagered']:.2f}, won ${data['won']:.2f})")
    
    # Calculate profitable days vs losing days
    profitable_days = sum(1 for d in daily_pnl.values() if d['net'] > 0)
    losing_days = sum(1 for d in daily_pnl.values() if d['net'] < 0)
    
    print()
    print("=" * 60)
    print("DAY-LEVEL SUMMARY")
    print("=" * 60)
    print(f"Profitable Days: {profitable_days}")
    print(f"Losing Days: {losing_days}")
    print(f"Break-even Days: {len(daily_pnl) - profitable_days - losing_days}")
    
    # Key insights
    print()
    print("=" * 60)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    insights = []
    
    if net_pnl < -50:
        insights.append("1. SIGNIFICANT LOSSES: Net loss exceeds $50. Risk management review needed.")
    
    if win_rate < 45:
        insights.append(f"2. LOW WIN RATE: {win_rate:.1f}% is below breakeven (~52.4% at -110 odds).")
    elif win_rate < 52:
        insights.append(f"2. MARGINAL WIN RATE: {win_rate:.1f}% is near breakeven but not profitable.")
    else:
        insights.append(f"2. GOOD WIN RATE: {win_rate:.1f}% should be profitable with proper odds.")
    
    if avg_win < avg_bet * 1.8:
        insights.append(f"3. POOR ODDS SELECTION: Average win (${avg_win:.2f}) vs bet (${avg_bet:.2f}) suggests -115 or worse juice.")
    
    if len(wagers) > 100 and net_pnl < 0:
        insights.append("4. VOLUME WITHOUT EDGE: High bet count with losses suggests over-trading or no predictive edge.")
    
    if profitable_days < losing_days:
        insights.append(f"5. CONSISTENCY ISSUE: More losing days ({losing_days}) than winning days ({profitable_days}).")
    
    for insight in insights:
        print(insight)
    
    if not insights:
        print("Performance appears solid. Continue current strategy.")
    
    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. Reduce bet frequency - focus on highest edge opportunities")
    print("2. Improve odds shopping - seek -105 or better when possible")
    print("3. Implement strict bankroll management (1-2% per bet max)")
    print("4. Review K-3 model audit: 0 bets may indicate over-conservatism")
    print("5. Wait for V9 model to accumulate 50+ bets for recalibration")
    print()
    
    return {
        'total_wagers': len(wagers),
        'total_wins': len(winnings),
        'net_pnl': net_pnl,
        'roi': net_pnl/abs(total_wagered)*100 if total_wagered else 0,
        'win_rate': win_rate,
        'avg_bet': avg_bet,
        'profitable_days': profitable_days,
        'losing_days': losing_days
    }

if __name__ == "__main__":
    stats = analyze_betting()
    
    # Save analysis
    with open('reports/betting_analysis.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("Analysis saved to reports/betting_analysis.json")
