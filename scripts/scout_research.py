"""
Autonomous Research Script — Sanity checks the model's bets against real-time news.
Uses DuckDuckGo for search and Local LLM for synthesis.
"""

import os
import sys
import json
import requests
from duckduckgo_search import DDGS

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.scout import perform_sanity_check

def run_scout_research():
    print("="*60)
    print("🕵️ SCOUT RESEARCH: REAL-TIME SANITY CHECK")
    print("="*60)

    # 1. Get today's bets from the API
    API_URL = "http://localhost:8000/api/predictions/today"
    API_KEY = "j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg"
    
    try:
        resp = requests.get(API_URL, headers={"X-API-Key": API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching today's bets: {e}")
        return

    predictions = data.get("predictions", [])
    bets = [p for p in predictions if p.get("verdict", "").startswith("Bet")]

    if not bets:
        print("No recommended bets found for today. Nothing to research.")
        return

    print(f"Found {len(bets)} bets to research. Offloading to DuckDuckGo + GPU...")

    results_summary = []

    with DDGS() as ddgs:
        for bet in bets:
            home = bet["game"]["home_team"]
            away = bet["game"]["away_team"]
            verdict = bet["verdict"]
            
            print(f"\n[Researching] {away} @ {home}...")
            
            # Perform search
            query = f"{away} vs {home} basketball injury news {data['date']}"
            search_results = ""
            try:
                results = ddgs.text(query, max_results=5)
                search_results = "\n".join([f"- {r['body']}" for r in results])
            except Exception as e:
                search_results = f"Search failed: {e}"

            # Pass to LLM for Sanity Check
            integrity_report = perform_sanity_check(home, away, verdict, search_results)
            print(f"INTEGRITY: {integrity_report}")
            
            results_summary.append({
                "game": f"{away} @ {home}",
                "report": integrity_report
            })

    print("\n" + "="*60)
    print("🏁 RESEARCH COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_scout_research()
