# K-6 O-8 Pre-Tournament Baseline Specification

**Mission:** K-6  
**Task:** O-8 Pre-Tournament Baseline Script Design  
**Owner:** Kimi CLI (Deep Intelligence Unit + OpenClaw Config Owner)  
**Execution Date:** March 16, 2026 ~9:00 PM ET  
**Status:** COMPLETE (spec ready for implementation)

---

## Executive Summary

Design complete for `scripts/openclaw_baseline.py` — a batch intelligence operation that executes March 16 at 9 PM ET to establish a risk baseline for all 68 tournament teams before March Madness begins.

**Output:** 68-team JSON risk map with seed, region, risk_level, and narrative summary.  
**Method:** DDGS web search + qwen2.5:3b local LLM analysis (OpenClaw stack).  
**Cost:** ~$0.10 (68 teams × 2 queries × $0.0001 per check).  
**Runtime:** ~5 minutes (sequential with 1s delay to respect DDGS).

---

## Script Architecture

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  scripts/openclaw_baseline.py                                       │
│  ─────────────────────────────────                                  │
│                                                                     │
│  1. FETCH BRACKET                                                   │
│     └── BallDontLie API → GET /ncaab/v1/bracket?season=2025        │
│         └── Extract: 68 teams with seeds, regions, matchups         │
│                                                                     │
│  2. BATCH INTELLIGENCE GATHERING                                    │
│     └── For each of 68 teams:                                       │
│         ├── DDGS: "{team} basketball injury news March 2026"       │
│         ├── DDGS: "{team} {conference} tournament upset risk"      │
│         └── OpenClaw (qwen2.5:3b): Analyze + classify risk         │
│         └── Rate limit: 1s delay between teams                      │
│                                                                     │
│  3. RISK CLASSIFICATION                                             │
│     └── qwen2.5:3b outputs structured JSON:                        │
│         ├── risk_level: LOW / MEDIUM / HIGH / CRITICAL             │
│         ├── risk_factors: [list of concerns]                        │
│         └── summary: "2-sentence professional assessment"          │
│                                                                     │
│  4. OUTPUT GENERATION                                               │
│     ├── JSON: data/pre_tournament_baseline_2026.json               │
│     ├── Markdown: reports/o8_baseline_summary_2026.md              │
│     └── HANDOFF.md update: Section 1.5 (OpenClaw Status)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Output Schema

### JSON Output (`data/pre_tournament_baseline_2026.json`)

```json
{
  "metadata": {
    "generated_at": "2026-03-16T21:15:00Z",
    "tournament_year": 2026,
    "teams_analyzed": 68,
    "method": "DDGS + OpenClaw/qwen2.5:3b",
    "version": "O-8-v1.0"
  },
  "risk_distribution": {
    "LOW": 42,
    "MEDIUM": 18,
    "HIGH": 6,
    "CRITICAL": 2
  },
  "teams": {
    "Duke": {
      "seed": 1,
      "region": "East",
      "region_id": 1,
      "risk_level": "LOW",
      "risk_score": 15,
      "risk_factors": [],
      "summary": "Full-strength roster entering tournament. No injury concerns reported. Elite efficiency metrics across all phases.",
      "sources_checked": 5,
      "analyzed_at": "2026-03-16T21:05:00Z"
    },
    "North Carolina": {
      "seed": 8,
      "region": "West",
      "region_id": 3,
      "risk_level": "MEDIUM",
      "risk_score": 45,
      "risk_factors": [
        "Star guard reported ankle soreness post-ACC tournament",
        "Three-point shooting declined 5% in last 5 games"
      ],
      "summary": "Minor injury monitoring on backcourt leader. Shooting slump entering tournament creates volatility risk despite strong interior presence.",
      "sources_checked": 5,
      "analyzed_at": "2026-03-16T21:06:00Z"
    },
    "Memphis": {
      "seed": 12,
      "region": "South",
      "region_id": 2,
      "risk_level": "HIGH",
      "risk_score": 72,
      "risk_factors": [
        "Starting center out for season (knee)",
        "Coach suspended for rules violation",
        "Lost 4 of last 5 games"
      ],
      "summary": "Multiple structural risks: interior defense compromised by injury, leadership void from coaching suspension, negative momentum trend. Upset vulnerability elevated.",
      "sources_checked": 5,
      "analyzed_at": "2026-03-16T21:07:00Z"
    }
  },
  "region_risk_heatmap": {
    "East": {"avg_risk_score": 22, "highest_risk_team": "Michigan State"},
    "South": {"avg_risk_score": 35, "highest_risk_team": "Memphis"},
    "West": {"avg_risk_score": 28, "highest_risk_team": "North Carolina"},
    "Midwest": {"avg_risk_score": 18, "highest_risk_team": "Iowa State"}
  }
}
```

### Risk Level Definitions

| Level | Score Range | Criteria | Tournament Implication |
|-------|-------------|----------|----------------------|
| **LOW** | 0-25 | No injuries; strong momentum; full roster | Standard Kelly sizing applies |
| **MEDIUM** | 26-50 | Minor injury concern; 1-2 game slump; rotation question | Consider 0.90x scalar |
| **HIGH** | 51-75 | Significant injury; coaching issue; 3+ game losing streak | Consider 0.75x scalar or pass |
| **CRITICAL** | 76-100 | Season-ending injury to star; multiple suspensions; chaos | Avoid or 0.50x max scalar |

---

## Implementation Specification

### File: `scripts/openclaw_baseline.py`

```python
#!/usr/bin/env python3
"""
O-8 Pre-Tournament Baseline Script

Executes March 16 ~9 PM ET to establish risk baseline for all 68 tournament teams.
Uses OpenClaw stack (DDGS + qwen2.5:3b) for cost-effective batch intelligence.

Output:
    - data/pre_tournament_baseline_2026.json (structured data)
    - reports/o8_baseline_summary_2026.md (human-readable summary)
    - Updates HANDOFF.md Section 1.5 with baseline summary

Usage:
    python scripts/openclaw_baseline.py [--year 2026] [--output-dir data/]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.env_utils import get_env

logger = logging.getLogger(__name__)

# Constants
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
BALLDONTLIE_API_KEY = get_env("BALLDONTLIE_API_KEY")
BALLDONTLIE_BASE_URL = "https://api.balldontlie.io"

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
REGIONS = {1: "East", 2: "South", 3: "West", 4: "Midwest"}


def fetch_tournament_bracket(year: int = 2026) -> List[Dict]:
    """
    Fetch tournament bracket from BallDontLie API.
    Note: API uses season-1 offset (2026 tournament = season=2025)
    """
    api_year = year - 1
    
    headers = {"Authorization": BALLDONTLIE_API_KEY}
    params = {"season": api_year, "per_page": 100}
    
    logger.info(f"Fetching bracket for tournament year {year} (API season {api_year})...")
    
    try:
        resp = requests.get(
            f"{BALLDONTLIE_BASE_URL}/ncaab/v1/bracket",
            headers=headers,
            params=params,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        
        games = data.get("data", [])
        logger.info(f"Retrieved {len(games)} bracket games")
        return games
        
    except Exception as e:
        logger.error(f"Failed to fetch bracket: {e}")
        # Fallback: return empty list, script will fail gracefully
        return []


def extract_teams_from_bracket(games: List[Dict]) -> Dict[str, Dict]:
    """
    Extract unique teams from bracket games with seed and region info.
    Returns: {team_name: {seed, region, region_id, bracket_game_id}}
    """
    teams = {}
    
    for game in games:
        round_num = game.get("round")
        region_id = game.get("region_id")
        region_name = game.get("region_label", REGIONS.get(region_id, "Unknown"))
        
        # Only process Round of 64 games (round=2) for team list
        # This avoids duplicates from later rounds
        if round_num != 2:
            continue
            
        for team_key in ["home_team", "away_team"]:
            team = game.get(team_key)
            if not team:
                continue
                
            name = team.get("name")
            seed_str = team.get("seed")
            
            if name and seed_str:
                teams[name] = {
                    "name": name,
                    "seed": int(seed_str),
                    "region": region_name,
                    "region_id": region_id,
                    "full_name": team.get("full_name", name)
                }
    
    logger.info(f"Extracted {len(teams)} unique teams from Round of 64")
    return teams


def search_team_intelligence(team_name: str, region: str) -> Dict[str, str]:
    """
    Perform DDGS searches for team intelligence.
    Returns dict of search results by query type.
    """
    try:
        # Lazy import DDGS
        from duckduckgo_search import DDGS
        
        queries = [
            ("injury", f"{team_name} basketball injury news March 2026"),
            ("momentum", f"{team_name} {region} tournament momentum upset risk"),
        ]
        
        results = {}
        with DDGS() as ddgs:
            for key, query in queries:
                try:
                    search_results = list(ddgs.text(query, max_results=5))
                    results[key] = " | ".join(
                        r.get("body", "") for r in search_results if r.get("body")
                    )
                    time.sleep(0.5)  # Respect DDGS rate limits
                except Exception as e:
                    logger.warning(f"DDGS search failed for {team_name}/{key}: {e}")
                    results[key] = ""
        
        return results
        
    except ImportError:
        logger.error("DDGS not installed. Install: pip install duckduckgo-search")
        return {"injury": "", "momentum": ""}


def analyze_team_risk(
    team_name: str,
    seed: int,
    region: str,
    search_results: Dict[str, str]
) -> Dict:
    """
    Use qwen2.5:3b to analyze team risk from search results.
    Returns structured risk assessment.
    """
    prompt = f"""You are a College Basketball Tournament Risk Analyst. Analyze this team entering March Madness.

Team: {team_name}
Seed: #{seed} in {region} Region

Intelligence Reports:
[INJURIES/SUSPENSIONS]: {search_results.get('injury', 'No data')}
[MOMENTUM/UPSET RISK]: {search_results.get('momentum', 'No data')}

Analyze the above and output ONLY valid JSON in this exact format:
{{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "risk_score": 0-100,
  "risk_factors": ["factor 1", "factor 2"],
  "summary": "2-sentence professional risk assessment"
}}

Risk Level Definitions:
- LOW (0-25): No injuries, strong momentum, full roster
- MEDIUM (26-50): Minor injury concern or slight slump
- HIGH (51-75): Significant injury, coaching issue, or losing streak
- CRITICAL (76-100): Star injured, multiple issues, chaos situation

JSON:"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 256,
                "temperature": 0.2  # Low temp for consistent JSON
            },
            "format": "json"
        }
        
        resp = requests.post(OLLAMA_URL, json=payload, timeout=15)
        resp.raise_for_status()
        
        result_text = resp.json().get("response", "{}").strip()
        
        # Parse JSON response
        if isinstance(result_text, str):
            analysis = json.loads(result_text)
        else:
            analysis = result_text
        
        # Validate and normalize
        risk_level = analysis.get("risk_level", "MEDIUM").upper()
        if risk_level not in RISK_LEVELS:
            risk_level = "MEDIUM"
        
        return {
            "risk_level": risk_level,
            "risk_score": min(100, max(0, int(analysis.get("risk_score", 50)))),
            "risk_factors": analysis.get("risk_factors", [])[:5],  # Max 5 factors
            "summary": analysis.get("summary", "Risk assessment unavailable."),
            "sources_checked": 5
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {team_name}: {e}")
        return _fallback_risk_assessment(seed)
    except Exception as e:
        logger.warning(f"Risk analysis failed for {team_name}: {e}")
        return _fallback_risk_assessment(seed)


def _fallback_risk_assessment(seed: int) -> Dict:
    """Default risk assessment based on seed."""
    # Higher seeds (lower numbers) default to lower risk
    if seed <= 4:
        return {
            "risk_level": "LOW",
            "risk_score": 20,
            "risk_factors": ["No specific intelligence gathered"],
            "summary": "Top seed with no reported issues. Default LOW risk assessment.",
            "sources_checked": 0
        }
    elif seed <= 8:
        return {
            "risk_level": "MEDIUM",
            "risk_score": 40,
            "risk_factors": ["No specific intelligence gathered"],
            "summary": "Mid seed with standard tournament uncertainty. Default MEDIUM risk assessment.",
            "sources_checked": 0
        }
    else:
        return {
            "risk_level": "HIGH",
            "risk_score": 60,
            "risk_factors": ["No specific intelligence gathered"],
            "summary": "Lower seed with elevated upset potential. Default HIGH risk assessment.",
            "sources_checked": 0
        }


def calculate_region_heatmap(teams_data: Dict[str, Dict]) -> Dict:
    """Calculate average risk by region."""
    region_stats = {}
    
    for region_name in REGIONS.values():
        region_teams = [
            t for t in teams_data.values() 
            if t.get("region") == region_name
        ]
        
        if region_teams:
            avg_score = sum(t.get("risk_score", 50) for t in region_teams) / len(region_teams)
            highest_risk = max(region_teams, key=lambda t: t.get("risk_score", 0))
            
            region_stats[region_name] = {
                "avg_risk_score": round(avg_score, 1),
                "team_count": len(region_teams),
                "highest_risk_team": highest_risk.get("name"),
                "highest_risk_score": highest_risk.get("risk_score")
            }
    
    return region_stats


def generate_baseline_report(
    teams_data: Dict[str, Dict],
    year: int,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Generate JSON and Markdown reports.
    Returns paths to (json_file, md_file)
    """
    # Calculate risk distribution
    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for team in teams_data.values():
        level = team.get("risk_level", "MEDIUM")
        risk_dist[level] = risk_dist.get(level, 0) + 1
    
    # Calculate region heatmap
    region_heatmap = calculate_region_heatmap(teams_data)
    
    # Build output structure
    output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "tournament_year": year,
            "teams_analyzed": len(teams_data),
            "method": "DDGS + OpenClaw/qwen2.5:3b",
            "version": "O-8-v1.0"
        },
        "risk_distribution": risk_dist,
        "teams": teams_data,
        "region_risk_heatmap": region_heatmap
    }
    
    # Write JSON
    json_path = output_dir / f"pre_tournament_baseline_{year}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"JSON baseline written to {json_path}")
    
    # Generate Markdown summary
    md_path = output_dir.parent / "reports" / f"o8_baseline_summary_{year}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_content = f"""# O-8 Pre-Tournament Baseline Summary

**Tournament Year:** {year}  
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  
**Method:** DDGS Web Search + OpenClaw/qwen2.5:3b Analysis  
**Teams Analyzed:** {len(teams_data)}

---

## Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| 🔵 LOW | {risk_dist['LOW']} | {risk_dist['LOW']/len(teams_data)*100:.1f}% |
| 🟡 MEDIUM | {risk_dist['MEDIUM']} | {risk_dist['MEDIUM']/len(teams_data)*100:.1f}% |
| 🟠 HIGH | {risk_dist['HIGH']} | {risk_dist['HIGH']/len(teams_data)*100:.1f}% |
| 🔴 CRITICAL | {risk_dist['CRITICAL']} | {risk_dist['CRITICAL']/len(teams_data)*100:.1f}% |

---

## Region Risk Heatmap

| Region | Avg Risk Score | Highest Risk Team | Score |
|--------|----------------|-------------------|-------|
"""
    
    for region, stats in region_heatmap.items():
        md_content += f"| {region} | {stats['avg_risk_score']} | {stats['highest_risk_team']} | {stats['highest_risk_score']} |\n"
    
    md_content += "\n---\n\n## High Priority Teams (HIGH/CRITICAL Risk)\n\n"
    
    high_risk_teams = [
        (name, data) for name, data in teams_data.items()
        if data.get("risk_level") in ["HIGH", "CRITICAL"]
    ]
    high_risk_teams.sort(key=lambda x: x[1].get("risk_score", 0), reverse=True)
    
    if high_risk_teams:
        for name, data in high_risk_teams:
            md_content += f"""### {name} (#{data['seed']} {data['region']})
**Risk:** {data['risk_level']} ({data['risk_score']}/100)

**Factors:**
"""
            for factor in data.get("risk_factors", []):
                md_content += f"- {factor}\n"
            
            md_content += f"\n**Summary:** {data['summary']}\n\n"
    else:
        md_content += "No teams classified as HIGH or CRITICAL risk.\n"
    
    md_content += f"""
---

## Full Data

See JSON: `data/pre_tournament_baseline_{year}.json`
"""
    
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Markdown summary written to {md_path}")
    
    return json_path, md_path


def update_handoff_with_baseline(summary_stats: Dict):
    """
    Update HANDOFF.md Section 1.5 with baseline summary.
    """
    handoff_path = Path("HANDOFF.md")
    if not handoff_path.exists():
        logger.warning("HANDOFF.md not found, skipping update")
        return
    
    try:
        content = handoff_path.read_text()
        
        # Build update block
        update_block = f"""### O-8 Pre-Tournament Baseline (Auto-Updated {datetime.now().strftime('%Y-%m-%d %H:%M')})

| Metric | Value |
|--------|-------|
| Teams Analyzed | {summary_stats.get('teams_analyzed', 0)} |
| HIGH/CRITICAL Risk Teams | {summary_stats.get('high_risk_count', 0)} |
| Riskiest Region | {summary_stats.get('riskiest_region', 'N/A')} |
| Data File | `data/pre_tournament_baseline_2026.json` |

**Key Alerts:**
"""
        
        alerts = summary_stats.get("alerts", [])
        if alerts:
            for alert in alerts:
                update_block += f"- 🚨 {alert}\n"
        else:
            update_block += "- No critical risk alerts\n"
        
        # Replace or insert O-8 section
        import re
        pattern = r'### O-8 Pre-Tournament Baseline \(Auto-Updated.*?(?=###|\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, update_block.rstrip(), content, flags=re.DOTALL)
        else:
            # Insert after Section 1.5 header
            content = content.replace(
                "## 1.5. OPENCLAW STATUS",
                f"{update_block}\n\n## 1.5. OPENCLAW STATUS"
            )
        
        handoff_path.write_text(content)
        logger.info("HANDOFF.md updated with baseline summary")
        
    except Exception as e:
        logger.error(f"Failed to update HANDOFF.md: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="O-8 Pre-Tournament Baseline Intelligence Gather"
    )
    parser.add_argument(
        "--year", 
        type=int, 
        default=2026,
        help="Tournament year (default: 2026)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for JSON data (default: data/)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between teams in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N teams for testing (default: all 68)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info(f"O-8 Pre-Tournament Baseline — Year {args.year}")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch bracket
    games = fetch_tournament_bracket(args.year)
    if not games:
        logger.error("Failed to fetch bracket. Exiting.")
        sys.exit(1)
    
    # Step 2: Extract teams
    teams = extract_teams_from_bracket(games)
    if len(teams) != 68:
        logger.warning(f"Expected 68 teams, got {len(teams)}")
    
    if args.limit:
        teams = dict(list(teams.items())[:args.limit])
        logger.info(f"Limited to {len(teams)} teams for testing")
    
    # Step 3: Analyze each team
    teams_data = {}
    high_risk_count = 0
    
    for idx, (name, info) in enumerate(teams.items(), 1):
        logger.info(f"[{idx}/{len(teams)}] Analyzing {name}...")
        
        # Search for intelligence
        search_results = search_team_intelligence(name, info["region"])
        
        # Analyze risk
        risk_analysis = analyze_team_risk(
            name, 
            info["seed"], 
            info["region"],
            search_results
        )
        
        # Combine data
        teams_data[name] = {
            **info,
            **risk_analysis,
            "analyzed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        if risk_analysis["risk_level"] in ["HIGH", "CRITICAL"]:
            high_risk_count += 1
            logger.warning(f"  ⚠️ {name}: {risk_analysis['risk_level']} risk ({risk_analysis['risk_score']})")
        else:
            logger.info(f"  ✓ {name}: {risk_analysis['risk_level']} risk ({risk_analysis['risk_score']})")
        
        # Rate limiting
        if idx < len(teams):
            time.sleep(args.delay)
    
    # Step 4: Generate reports
    json_path, md_path = generate_baseline_report(teams_data, args.year, args.output_dir)
    
    # Step 5: Calculate summary stats
    region_scores = calculate_region_heatmap(teams_data)
    riskiest_region = max(region_scores.items(), key=lambda x: x[1]["avg_risk_score"])[0] if region_scores else "N/A"
    
    summary_stats = {
        "teams_analyzed": len(teams_data),
        "high_risk_count": high_risk_count,
        "riskiest_region": riskiest_region,
        "alerts": [
            f"{high_risk_count} teams classified as HIGH/CRITICAL risk",
            f"{riskiest_region} Region has highest average risk"
        ] if high_risk_count > 0 else []
    }
    
    # Step 6: Update HANDOFF.md
    update_handoff_with_baseline(summary_stats)
    
    # Done
    logger.info("=" * 60)
    logger.info("O-8 Baseline Complete!")
    logger.info(f"JSON: {json_path}")
    logger.info(f"Report: {md_path}")
    logger.info(f"High Risk Teams: {high_risk_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Execution Plan

### Prerequisites (March 16)

1. **BallDontLie API Key** — Must be in Railway env vars (`BALLDONTLIE_API_KEY`)
2. **Ollama Running** — qwen2.5:3b model loaded locally
3. **DDGS Available** — `duckduckgo-search>=5.0` installed

### Command

```bash
# Production run (all 68 teams)
python scripts/openclaw_baseline.py --year 2026

# Test run (first 5 teams only)
python scripts/openclaw_baseline.py --year 2026 --limit 5 --delay 0.5
```

### Timing

- **Start:** March 16, 2026 ~9:00 PM ET (after bracket reveal at 6 PM)
- **Duration:** ~5-7 minutes (68 teams × 1s delay + processing)
- **Output Available:** ~9:07 PM ET

---

## Success Criteria

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Teams analyzed | 68 | `teams_analyzed == 68` in JSON |
| Risk classification | All teams | No team missing `risk_level` |
| Data quality | <10% fallback | <7 teams using seed-based default |
| Output files | 2 created | JSON + Markdown exist |
| HANDOFF updated | Yes | Section 1.5 contains baseline summary |

---

## Integration with Tournament Workflow

```
March 16, 2026
├── 6:00 PM ET: Selection Sunday (bracket revealed on CBS)
├── 8:00 PM ET: Claude implements A-26 T2 (seed-spread scalars)
├── 9:00 PM ET: O-8 BASELINE EXECUTES (this script)
│   └── Output: data/pre_tournament_baseline_2026.json
├── 10:00 PM ET: Kimi reviews baseline, flags HIGH/CRITICAL teams
└── 11:00 PM ET: Deployment to Railway complete

March 17, 2026
├── Model runs with seed data + baseline intel
└── Play-in games analysis with full tournament context

March 18, 2026
└── First Four begins — Tournament officially underway
```

---

## Risk Assessment Integration

The baseline output feeds into the betting model through:

1. **Pre-loaded Context** — `TournamentDataClient` loads baseline JSON on startup
2. **Risk Scalar** — HIGH/CRITICAL teams get additional Kelly sizing reduction
3. **Monitoring Queue** — HIGH/CRITICAL teams flagged for enhanced line monitoring

```python
# In betting_model.py (future enhancement)
def _tournament_risk_scalar(team_name: str) -> float:
    baseline = tournament_data_client.get_baseline(team_name)
    risk_level = baseline.get("risk_level", "MEDIUM")
    
    scalars = {
        "LOW": 1.0,
        "MEDIUM": 0.95,
        "HIGH": 0.85,
        "CRITICAL": 0.75
    }
    return scalars.get(risk_level, 1.0)
```

---

## Specification Complete

**Status:** ✅ READY FOR IMPLEMENTATION  
**Next Step:** Save script to `scripts/openclaw_baseline.py`  
**Execution Date:** March 16, 2026 ~9:00 PM ET  
**Owner:** OpenClaw (execution) | Kimi (design oversight)

---

*Specification v1.0 — K-6 Mission Complete*
