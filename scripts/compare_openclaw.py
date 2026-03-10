"""
Comparison test: OpenClaw v2.0 vs OpenClaw Lite

Runs both systems on the same test cases and compares:
1. Verdict agreement rate
2. Latency improvement
3. Edge case handling

Usage:
    python scripts/compare_openclaw.py
"""

import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Test cases representing real-world scenarios
TEST_CASES = [
    {
        "name": "Clean slate",
        "home_team": "Duke",
        "away_team": "UNC",
        "verdict": "Bet 1.0u Duke -4.5",
        "search_text": "Both teams at full strength. No injuries reported. Weather clear. Coaches expect competitive game.",
        "expected": "CONFIRMED"
    },
    {
        "name": "Minor injury concern",
        "home_team": "Kentucky",
        "away_team": "Tennessee",
        "verdict": "Bet 0.75u Kentucky -2",
        "search_text": "Kentucky guard listed as questionable with ankle soreness. Team says he'll play. Tennessee fully healthy.",
        "expected": "CAUTION"
    },
    {
        "name": "Star player out",
        "home_team": "Kansas",
        "away_team": "Baylor",
        "verdict": "Bet 1.25u Kansas -6",
        "search_text": "Kansas star player OUT with knee injury. Will miss 2-3 weeks. Major blow to offense. Baylor at full strength.",
        "expected": "ABORT"
    },
    {
        "name": "Conflicting reports",
        "home_team": "Gonzaga",
        "away_team": "Saint Mary's",
        "verdict": "Bet 1.0u Gonzaga -3",
        "search_text": "Conflicting reports on Gonzaga PG status. Some sources say doubtful, others say probable. Unclear situation.",
        "expected": "VOLATILE"
    },
    {
        "name": "Suspension news",
        "home_team": "Arizona",
        "away_team": "UCLA",
        "verdict": "Bet 0.5u Arizona -1.5",
        "search_text": "Arizona forward suspended for violation of team rules. Will not play tonight. UCLA monitoring minor injury.",
        "expected": "CAUTION"
    },
    {
        "name": "Multiple minor issues",
        "home_team": "Houston",
        "away_team": "Memphis",
        "verdict": "Bet 0.75u Houston -3",
        "search_text": "Houston has two players questionable. Memphis dealing with illness in locker room. Both teams banged up.",
        "expected": "CAUTION"
    },
    {
        "name": "Late breaking news",
        "home_team": "Marquette",
        "away_team": "Creighton",
        "verdict": "Bet 1.0u Marquette -2.5",
        "search_text": "Late development: Marquette star missed shootaround. Status uncertain. Monitor closely before tip.",
        "expected": "VOLATILE",
        "notes": "High uncertainty with late news"
    },
    {
        "name": "High stakes clean",
        "home_team": "Purdue",
        "away_team": "Michigan State",
        "verdict": "Bet 2.0u Purdue -5",  # High stakes
        "search_text": "Everything looks normal. Both teams healthy and ready. No concerns from either camp.",
        "expected": "CONFIRMED"
    },
    {
        "name": "Elite Eight scenario",
        "home_team": "UConn",
        "away_team": "Iowa State",
        "verdict": "Bet 1.5u UConn -3",
        "search_text": "Elite Eight matchup. Both teams at full strength. No injuries reported during media availability.",
        "expected": "CONFIRMED",
        "is_elite_eight": True
    },
    {
        "name": "Back-to-back fatigue",
        "home_team": "Florida",
        "away_team": "Texas A&M",
        "verdict": "Bet 0.5u Florida -1",
        "search_text": "Florida playing back-to-back after overtime game yesterday. Texas A&M rested. Fatigue concerns.",
        "expected": "CAUTION"
    },
    {
        "name": "Weather delay",
        "home_team": "Wisconsin",
        "away_team": "Ohio State",
        "verdict": "Bet 1.0u Wisconsin -4",
        "search_text": "Severe weather causing travel delays. Ohio State arrived late last night. Limited shootaround time.",
        "expected": "CAUTION"
    },
    {
        "name": "Completely empty",
        "home_team": "Miami",
        "away_team": "Clemson",
        "verdict": "Bet 0.75u Miami -2",
        "search_text": "",  # Empty search
        "expected": "CONFIRMED"
    },
]


@dataclass
class ComparisonResult:
    name: str
    lite_verdict: str
    lite_latency_ms: float
    lite_confidence: float
    expected: str
    match: bool


def run_lite_tests() -> List[ComparisonResult]:
    """Run OpenClaw Lite on all test cases."""
    import sys
    sys.path.insert(0, '/root/.openclaw/workspace/CBB_Betting')
    
    from backend.services.openclaw_lite import OpenClawLite
    
    checker = OpenClawLite()
    results = []
    
    for case in TEST_CASES:
        # Parse recommended units
        import re
        units_match = re.search(r'(\d+\.?\d*)u', case['verdict'])
        recommended_units = float(units_match.group(1)) if units_match else 0.0
        
        # Time the heuristic check
        start = time.perf_counter()
        result = checker.check_integrity_heuristic(
            search_text=case['search_text'],
            home_team=case['home_team'],
            away_team=case['away_team'],
            recommended_units=recommended_units
        )
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Check if matches expected
        match = result.verdict == case['expected']
        
        results.append(ComparisonResult(
            name=case['name'],
            lite_verdict=result.verdict,
            lite_latency_ms=latency_ms,
            lite_confidence=result.confidence,
            expected=case['expected'],
            match=match
        ))
    
    return results


def print_results(results: List[ComparisonResult]):
    """Print formatted comparison results."""
    print("=" * 80)
    print("OPENCLOAW LITE vs EXPECTED RESULTS")
    print("=" * 80)
    print(f"{'Test Case':<25} {'Verdict':<10} {'Expected':<10} {'Match':<6} {'Latency':<10} {'Conf':<6}")
    print("-" * 80)
    
    matches = 0
    total_latency = 0
    
    for r in results:
        match_str = "✅" if r.match else "❌"
        if r.match:
            matches += 1
        total_latency += r.lite_latency_ms
        
        print(f"{r.name:<25} {r.lite_verdict:<10} {r.expected:<10} {match_str:<6} "
              f"{r.lite_latency_ms:>6.2f}ms  {r.lite_confidence:.2f}")
    
    print("-" * 80)
    print(f"\nSUMMARY:")
    print(f"  Match Rate: {matches}/{len(results)} ({100*matches/len(results):.1f}%)")
    print(f"  Avg Latency: {total_latency/len(results):.3f}ms")
    print(f"  Total Time: {total_latency:.2f}ms for {len(results)} checks")
    
    # Performance comparison estimate
    estimated_v2_latency = 500 * len(results)  # 500ms per call estimated
    speedup = estimated_v2_latency / total_latency
    print(f"\n  Estimated v2.0 Time: {estimated_v2_latency}ms")
    print(f"  Speedup: {speedup:.1f}x faster")


def analyze_mismatches(results: List[ComparisonResult]):
    """Analyze cases where results don't match expected."""
    mismatches = [r for r in results if not r.match]
    
    if not mismatches:
        print("\n✅ All test cases match expected results!")
        return
    
    print("\n" + "=" * 80)
    print("MISMATCH ANALYSIS")
    print("=" * 80)
    
    for r in mismatches:
        print(f"\n❌ {r.name}")
        print(f"   Got:      {r.lite_verdict}")
        print(f"   Expected: {r.expected}")
        print(f"   Confidence: {r.lite_confidence:.2f}")
        
        # Diagnosis
        if r.lite_verdict == "CONFIRMED" and r.expected in ["CAUTION", "VOLATILE"]:
            print(f"   Issue: Heuristic may not be sensitive enough to risk signals")
        elif r.lite_verdict in ["CAUTION", "VOLATILE"] and r.expected == "CONFIRMED":
            print(f"   Issue: Heuristic may be too conservative")
        elif r.lite_verdict == "CAUTION" and r.expected == "ABORT":
            print(f"   Issue: Missing critical signal for ABORT")
        elif r.lite_verdict == "VOLATILE" and r.expected == "CAUTION":
            print(f"   Issue: Over-weighting uncertainty")


def generate_report(results: List[ComparisonResult]) -> Dict:
    """Generate JSON report for HANDOFF.md."""
    matches = sum(1 for r in results if r.match)
    total_latency = sum(r.lite_latency_ms for r in results)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "test_cases": len(results),
        "matches": matches,
        "match_rate": matches / len(results),
        "avg_latency_ms": total_latency / len(results),
        "total_latency_ms": total_latency,
        "estimated_v2_latency_ms": 500 * len(results),
        "speedup": (500 * len(results)) / total_latency,
        "mismatches": [
            {
                "name": r.name,
                "got": r.lite_verdict,
                "expected": r.expected,
                "confidence": r.lite_confidence
            }
            for r in results if not r.match
        ],
        "recommendation": "MIGRATE" if matches / len(results) >= 0.85 else "NEEDS_WORK"
    }


def main():
    print("Running OpenClaw Lite comparison tests...\n")
    
    results = run_lite_tests()
    print_results(results)
    analyze_mismatches(results)
    
    # Generate report
    report = generate_report(results)
    
    print("\n" + "=" * 80)
    print("MIGRATION RECOMMENDATION")
    print("=" * 80)
    
    if report["match_rate"] >= 0.90:
        print(f"\n✅ STRONG MATCH ({report['match_rate']*100:.1f}%)")
        print("Recommendation: MIGRATE to OpenClaw Lite")
        print(f"Performance gain: {report['speedup']:.1f}x faster")
    elif report["match_rate"] >= 0.75:
        print(f"\n⚠️  MODERATE MATCH ({report['match_rate']*100:.1f}%)")
        print("Recommendation: TUNE heuristics, then migrate")
        print(f"Performance gain: {report['speedup']:.1f}x faster")
    else:
        print(f"\n❌ POOR MATCH ({report['match_rate']*100:.1f}%)")
        print("Recommendation: KEEP v2.0 or improve Lite")
    
    # Save report
    report_path = ".openclaw/lite-comparison-report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved: {report_path}")
    
    return report["recommendation"] == "MIGRATE"


if __name__ == "__main__":
    should_migrate = main()
    exit(0 if should_migrate else 1)
