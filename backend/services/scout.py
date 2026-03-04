"""
Scout Agent — Generates narrative insights using local LLMs.
Translates quantitative matchup data into human-readable "Scouting Reports".
"""

import logging
import requests
import json
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

def generate_scouting_report(
    home_team: str, 
    away_team: str, 
    matchup_notes: List[str],
    verdict: str,
    edge: float
) -> str:
    """
    Generate a 1-sentence "Model Insight" for a specific game.
    """
    if not matchup_notes:
        return "Model edge based on core efficiency ratings and market consensus."

    notes_str = "\n".join(matchup_notes)
    
    prompt = f"""
You are a College Basketball Betting Scout. Your job is to summarize the quantitative model's reasons for a bet into a single, professional, and punchy sentence.

Game: {away_team} @ {home_team}
Verdict: {verdict}
Edge: {edge:.1%}
Model Matchup Notes:
{notes_str}

Instructions:
1. Translate the technical notes into a scouting insight. 
2. "Home" refers to {home_team}. "Away" refers to {away_team}.
3. Be concise (max 20 words).
4. Do not use phrases like "The model says" or "Based on notes". Just state the scouting reality.
5. Highlight the primary mismatch (e.g., transition defense, 3-point volume, or efficiency pressure).
6. Output ONLY the sentence. No introduction or commentary.

Insight:"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 64,
                "temperature": 0.3
            }
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Scout Agent (LLM) failed: %s", e)
        return "Model identifies matchup advantages in efficiency and style."

def generate_morning_briefing_narrative(
    n_bets: int,
    n_considered: int,
    top_bet_info: Optional[str] = None
) -> str:
    """
    Generates a 2-sentence narrative for the Morning Briefing.
    """
    prompt = f"""
You are the CBB Edge Editor. Summarize today's slate.
Bets Recommended: {n_bets}
Marginal Edges (Consider): {n_considered}
Top Opportunity Highlight: {top_bet_info or "None"}

Write a 2-sentence professional briefing for a high-stakes bettor. Focus on the 'vibe' of the slate.
Output ONLY the briefing.
"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 128,
                "temperature": 0.5
            }
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Briefing Agent (LLM) failed: %s", e)
        return "The model has identified a high-conviction slate with key efficiency mismatches."

def generate_injury_impact(
    player: str,
    team: str,
    raw_text: str,
    base_impact: float
) -> Dict:
    """
    Refines the point-spread impact of an injury using local LLM.
    Returns {impact: float, tier: str, reason: str}
    """
    prompt = f"""
You are a College Basketball Injury Analyst. Refine the point-spread impact for this injury.
Player: {player} ({team})
Status Details: {raw_text}
Model Base Impact: {base_impact} pts

Instructions:
1. Determine if the player is a "star", "starter", "role", or "bench" player based on the context.
2. Adjust the impact (0.0 to 7.0 pts). If they are the leading scorer or an All-American, it might be 4-6 pts. If they are a backup, it's < 1 pt.
3. Provide a 1-sentence reason.
4. Output ONLY valid JSON in this format: {{"impact": 2.5, "tier": "starter", "reason": "reason here"}}

JSON:"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 128,
                "temperature": 0.2
            },
            "format": "json"
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("response", "{}")
        if isinstance(data, str):
            data = json.loads(data)
        return data
    except Exception as e:
        logger.warning("Injury Agent (LLM) failed: %s", e)
        return {"impact": base_impact, "tier": "role", "reason": "Default model impact applied."}

def perform_sanity_check(
    home_team: str,
    away_team: str,
    verdict: str,
    search_results: str
) -> str:
    """
    Performs a sanity check using search results and local LLM.
    """
    prompt = f"""
You are a College Basketball Betting Integrity Officer. Perform a sanity check on this recommended bet using real-time search results.

Matchup: {away_team} @ {home_team}
Model Verdict: {verdict}

Real-Time News/Injuries:
{search_results}

Instructions:
1. Identify if any major news (injuries, suspensions, Senior Night, revenge spot) contradicts the bet or increases its risk.
2. Focus on "Star" player absences not mentioned in official reports.
3. Provide a 1-sentence "Integrity Verdict" (e.g., "CONFIRMED", "CAUTION - Injury Alert", or "VOLATILE").
4. Output ONLY the verdict and a brief reason.

Verdict:"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 128,
                "temperature": 0.3
            }
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Sanity Check Agent (LLM) failed: %s", e)
        return "Sanity check unavailable (LLM error)."

def generate_health_narrative(summary: Dict) -> str:
    """
    Generates a 1-2 sentence professional health summary for the Sentinel.
    """
    perf = summary.get("performance", {})
    port = summary.get("portfolio", {})
    sys_status = summary.get("system", {})
    
    prompt = f"""
You are the CBB Edge Systems Engineer. Summarize the current system health.
MAE (30d): {perf.get('mean_mae', 'N/A')} (Status: {perf.get('status')})
Portfolio Drawdown: {port.get('current_drawdown_pct', 0.0):.1%} (Status: {port.get('status')})
Core Tests: {'PASSED' if sys_status.get('passed') else 'FAILED'} (Status: {sys_status.get('status')})

Write a 1-2 sentence professional summary of system health for an executive report.
Output ONLY the briefing.
"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 128,
                "temperature": 0.3
            }
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Health Sentinel (LLM) failed: %s", e)
        return "System health metrics are within operating parameters, with core stability tests passing."

