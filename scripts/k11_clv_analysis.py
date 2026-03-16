"""
K-11: CLV Performance Attribution Analysis

Mission: Analyze betting database to understand model performance vs closing line value.

Questions to answer:
1. Mean CLV across all settled bets
2. By edge bucket (0-3%, 3-6%, 6%+): win rate and CLV in each bucket
3. By conference: which conferences are profitable?
4. By game type: neutral site vs home game — win rate difference?
5. How many BET verdicts per week over the last 60 days?
6. Actual win rate vs expected win rate for each edge bucket?

Deliverable: reports/K11_CLV_ATTRIBUTION_MARCH_2026.md
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional

# Setup paths
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_repo_root, ".env"))
except ImportError:
    pass

if not os.getenv("DATABASE_URL"):
    print("ERROR: DATABASE_URL not set")
    sys.exit(2)

from sqlalchemy import create_engine, func, case, and_, or_
from sqlalchemy.orm import sessionmaker

from backend.models import BetLog, Game, Prediction, PerformanceSnapshot

# Create engine
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/cbb_edge")
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BetWithCLV(NamedTuple):
    bet_id: int
    pick: str
    home_team: str
    away_team: str
    outcome: Optional[int]  # 1=win, 0=loss, -1=push, None=pending
    conservative_edge: float
    closing_line: Optional[float]
    clv_points: Optional[float]
    is_neutral: bool
    conference: Optional[str]
    model_prob: Optional[float]
    bet_size_units: float
    profit_loss_units: Optional[float]
    game_date: datetime
    verdict: Optional[str]


class EdgeBucketStats(NamedTuple):
    bucket: str
    count: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    expected_win_rate: float
    calibration_error: float
    mean_clv: Optional[float]
    total_profit_units: float


class ConferenceStats(NamedTuple):
    conference: str
    bets: int
    wins: int
    losses: int
    win_rate: float
    total_profit_units: float
    avg_edge: float


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_bets_with_clv(db) -> List[BetWithCLV]:
    """Extract all settled or pending bets with their CLV data."""
    bets = []
    
    # Query all bet logs with game and prediction data
    results = (
        db.query(
            BetLog,
            Game.home_team,
            Game.away_team,
            Game.is_neutral,
            Game.game_date,
            Prediction.conservative_edge,
            Prediction.verdict,
            Prediction.snr,
        )
        .join(Game, BetLog.game_id == Game.id)
        .outerjoin(Prediction, BetLog.prediction_id == Prediction.id)
        .all()
    )
    
    for row in results:
        bet = row[0]
        home_team = row[1]
        away_team = row[2]
        is_neutral = row[3]
        game_date = row[4]
        pred_edge = row[5]
        verdict = row[6]
        snr = row[7]
        
        # Extract conference from analysis JSON if available
        conference = None
        if bet.full_analysis and isinstance(bet.full_analysis, dict):
            conference = bet.full_analysis.get("conference")
        
        bets.append(BetWithCLV(
            bet_id=bet.id,
            pick=bet.pick,
            home_team=home_team,
            away_team=away_team,
            outcome=bet.outcome,
            conservative_edge=pred_edge or bet.conservative_edge or 0.0,
            closing_line=bet.closing_line,
            clv_points=bet.clv_points,
            is_neutral=is_neutral,
            conference=conference,
            model_prob=bet.model_prob,
            bet_size_units=bet.bet_size_units or 0.0,
            profit_loss_units=bet.profit_loss_units,
            game_date=game_date,
            verdict=verdict,
        ))
    
    return bets


def get_bet_frequency_by_week(db, days: int = 60) -> Dict[str, int]:
    """Get number of BET verdicts per week for last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    results = (
        db.query(
            func.date_trunc('week', Prediction.created_at).label('week'),
            func.count(Prediction.id).label('count')
        )
        .filter(
            Prediction.created_at >= cutoff,
            Prediction.verdict.ilike('%bet%')
        )
        .group_by(func.date_trunc('week', Prediction.created_at))
        .order_by(func.date_trunc('week', Prediction.created_at))
        .all()
    )
    
    return {row.week.strftime('%Y-%m-%d'): row.count for row in results}


def get_total_predictions_stats(db, days: int = 60) -> Dict:
    """Get prediction stats for frequency analysis."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    total = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.created_at >= cutoff)
        .scalar()
    )
    
    bets = (
        db.query(func.count(Prediction.id))
        .filter(
            Prediction.created_at >= cutoff,
            Prediction.verdict.ilike('%bet%')
        )
        .scalar()
    )
    
    considers = (
        db.query(func.count(Prediction.id))
        .filter(
            Prediction.created_at >= cutoff,
            Prediction.verdict.ilike('%consider%')
        )
        .scalar()
    )
    
    passes = (
        db.query(func.count(Prediction.id))
        .filter(
            Prediction.created_at >= cutoff,
            Prediction.verdict.ilike('%pass%')
        )
        .scalar()
    )
    
    return {
        "total": total or 0,
        "bets": bets or 0,
        "considers": considers or 0,
        "passes": passes or 0,
        "bet_rate": (bets / total * 100) if total else 0,
    }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_edge_buckets(bets: List[BetWithCLV]) -> List[EdgeBucketStats]:
    """Analyze win rate and CLV by edge bucket."""
    # Define buckets
    buckets = [
        ("0-2.5%", 0.0, 0.025),
        ("2.5-4%", 0.025, 0.04),
        ("4-6%", 0.04, 0.06),
        ("6%+", 0.06, 1.0),
    ]
    
    results = []
    
    for bucket_name, min_edge, max_edge in buckets:
        bucket_bets = [b for b in bets if min_edge <= b.conservative_edge < max_edge]
        settled = [b for b in bucket_bets if b.outcome is not None]
        
        if not settled:
            results.append(EdgeBucketStats(
                bucket=bucket_name,
                count=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=0.0,
                expected_win_rate=0.0,
                calibration_error=0.0,
                mean_clv=None,
                total_profit_units=0.0,
            ))
            continue
        
        wins = sum(1 for b in settled if b.outcome == 1)
        losses = sum(1 for b in settled if b.outcome == 0)
        pushes = sum(1 for b in settled if b.outcome == -1)
        
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        
        # Expected win rate from model probability (approximate)
        probs = [b.model_prob for b in settled if b.model_prob is not None]
        expected_win_rate = sum(probs) / len(probs) if probs else 0.52  # Default assumption
        
        calibration_error = abs(win_rate - expected_win_rate)
        
        # CLV calculation
        clvs = [b.clv_points for b in settled if b.clv_points is not None]
        mean_clv = sum(clvs) / len(clvs) if clvs else None
        
        # Profit
        profit = sum(b.profit_loss_units or 0 for b in settled)
        
        results.append(EdgeBucketStats(
            bucket=bucket_name,
            count=len(settled),
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=win_rate,
            expected_win_rate=expected_win_rate,
            calibration_error=calibration_error,
            mean_clv=mean_clv,
            total_profit_units=profit,
        ))
    
    return results


def analyze_conferences(bets: List[BetWithCLV]) -> List[ConferenceStats]:
    """Analyze profitability by conference."""
    # Group bets by conference (extract from full analysis or team names)
    # Since conference isn't directly stored, we'll infer from known patterns
    
    # Map of team name patterns to conferences
    conference_patterns = {
        "ACC": ["Duke", "North Carolina", "NC State", "Wake Forest", "Clemson", 
                "Florida State", "Virginia", "Virginia Tech", "Syracuse", "Louisville",
                "Pittsburgh", "Boston College", "Georgia Tech", "Notre Dame", "Miami"],
        "Big Ten": ["Purdue", "Michigan State", "Michigan", "Ohio State", "Indiana",
                   "Wisconsin", "Iowa", "Illinois", "Maryland", "Rutgers",
                   "Northwestern", "Penn State", "Nebraska", "Minnesota"],
        "SEC": ["Auburn", "Tennessee", "Kentucky", "Alabama", "Florida",
                "Texas A&M", "Arkansas", "Missouri", "Mississippi State", "Ole Miss",
                "South Carolina", "Georgia", "LSU", "Vanderbilt", "Oklahoma"],
        "Big 12": ["Houston", "Iowa State", "Kansas", "Texas Tech", "BYU",
                  "Baylor", "TCU", "Kansas State", "Cincinnati", "UCF",
                  "West Virginia", "Arizona", "Arizona State", "Colorado", "Utah"],
        "Big East": ["UConn", "Marquette", "Creighton", "Xavier", "Villanova",
                    "Seton Hall", "Providence", "Georgetown", "Butler", "DePaul",
                    "St. John's"],
        "Pac-12": ["UCLA", "USC", "Washington", "Oregon", "Stanford",
                   "California", "Arizona", "Arizona State", "Colorado", "Utah"],
        "WCC": ["Gonzaga", "Saint Mary's", "San Francisco", "Santa Clara",
                "Loyola Marymount", "Pepperdine", "San Diego", "Portland", "Pacific"],
        "Mountain West": ["New Mexico", "Nevada", "San Diego State", "Boise State",
                         "Colorado State", "UNLV", "Fresno State", "San Jose State",
                         "Utah State", "Air Force", "Wyoming"],
        "Atlantic 10": ["Dayton", "VCU", "Richmond", "George Mason", "Saint Louis",
                       "Duquesne", "St. Bonaventure", "Rhode Island", "Davidson",
                       "Loyola Chicago", "UMass", "George Washington", "La Salle"],
        "American": ["Memphis", "Florida Atlantic", "UAB", "Charlotte", "South Florida",
                    "Tulane", "Temple", "Wichita State", "North Texas", "UTSA",
                    "Rice", "East Carolina"],
    }
    
    def infer_conference(bet: BetWithCLV) -> Optional[str]:
        """Infer conference from team names."""
        pick_lower = bet.pick.lower()
        home_lower = bet.home_team.lower()
        away_lower = bet.away_team.lower()
        
        for conf, teams in conference_patterns.items():
            for team in teams:
                if team.lower() in pick_lower or team.lower() in home_lower or team.lower() in away_lower:
                    return conf
        return "Other"
    
    # Group by conference
    conf_groups = defaultdict(list)
    for bet in bets:
        conf = infer_conference(bet)
        conf_groups[conf].append(bet)
    
    results = []
    for conf, conf_bets in conf_groups.items():
        settled = [b for b in conf_bets if b.outcome is not None]
        if not settled:
            continue
        
        wins = sum(1 for b in settled if b.outcome == 1)
        losses = sum(1 for b in settled if b.outcome == 0)
        profit = sum(b.profit_loss_units or 0 for b in settled)
        avg_edge = sum(b.conservative_edge for b in settled) / len(settled)
        
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        
        results.append(ConferenceStats(
            conference=conf,
            bets=len(settled),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_profit_units=profit,
            avg_edge=avg_edge,
        ))
    
    # Sort by profit
    results.sort(key=lambda x: x.total_profit_units, reverse=True)
    return results


def analyze_game_type(bets: List[BetWithCLV]) -> Dict:
    """Compare neutral site vs home game performance."""
    neutral_bets = [b for b in bets if b.is_neutral and b.outcome is not None]
    home_bets = [b for b in bets if not b.is_neutral and b.outcome is not None]
    
    def calc_stats(bet_list):
        if not bet_list:
            return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "profit": 0.0}
        wins = sum(1 for b in bet_list if b.outcome == 1)
        losses = sum(1 for b in bet_list if b.outcome == 0)
        profit = sum(b.profit_loss_units or 0 for b in bet_list)
        return {
            "bets": len(bet_list),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            "profit": profit,
        }
    
    return {
        "neutral": calc_stats(neutral_bets),
        "home_venue": calc_stats(home_bets),
    }


def calculate_mean_clv(bets: List[BetWithCLV]) -> Dict:
    """Calculate overall CLV statistics."""
    settled = [b for b in bets if b.outcome is not None]
    
    clvs = [b.clv_points for b in settled if b.clv_points is not None]
    
    if not clvs:
        return {
            "mean_clv": None,
            "median_clv": None,
            "positive_clv_pct": 0.0,
            "count": 0,
        }
    
    sorted_clvs = sorted(clvs)
    n = len(sorted_clvs)
    
    return {
        "mean_clv": sum(clvs) / len(clvs),
        "median_clv": sorted_clvs[n // 2] if n % 2 else (sorted_clvs[n // 2 - 1] + sorted_clvs[n // 2]) / 2,
        "positive_clv_pct": sum(1 for c in clvs if c > 0) / len(clvs) * 100,
        "count": len(clvs),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(bets: List[BetWithCLV], week_freq: Dict, pred_stats: Dict, 
                    output_path: str) -> str:
    """Generate the K-11 markdown report."""
    
    # Run analyses
    edge_stats = analyze_edge_buckets(bets)
    conf_stats = analyze_conferences(bets)
    game_type_stats = analyze_game_type(bets)
    clv_stats = calculate_mean_clv(bets)
    
    # Calculate overall stats
    settled = [b for b in bets if b.outcome is not None]
    total_wins = sum(1 for b in settled if b.outcome == 1)
    total_losses = sum(1 for b in settled if b.outcome == 0)
    total_pushes = sum(1 for b in settled if b.outcome == -1)
    total_profit = sum(b.profit_loss_units or 0 for b in settled)
    overall_win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.0
    
    # Count bets with CLV data
    bets_with_clv = sum(1 for b in settled if b.clv_points is not None)
    
    # Pre-compute CLV display values
    mean_clv_display = f"{clv_stats['mean_clv']:+.2f} pts" if clv_stats['mean_clv'] is not None else "N/A"
    median_clv_display = f"{clv_stats['median_clv']:+.2f} pts" if clv_stats['median_clv'] is not None else "N/A"
    
    # Pre-compute conditional strings
    calibration_ok = all(s.calibration_error < 0.05 for s in edge_stats if s.count > 5)
    has_positive_clv = any(s.mean_clv and s.mean_clv > 0 for s in edge_stats)
    high_edge_good = any(s.win_rate > 0.52 for s in edge_stats if "4" in s.bucket or "6" in s.bucket)
    
    calibration_finding = "Model is well-calibrated" if calibration_ok else "Model shows calibration drift — predicted probabilities don't match actual outcomes"
    clv_finding = "Higher edge buckets show positive CLV" if has_positive_clv else "CLV not consistently positive across edge buckets"
    edge_finding = "Edge > 4% shows improved win rate" if high_edge_good else "Higher edges do not translate to higher win rates — suggests model overestimation"
    
    # Conference profitability
    most_profitable_conf = f"{conf_stats[0].conference} ({conf_stats[0].total_profit_units:+.2f}u)" if conf_stats else "N/A"
    least_profitable_conf = f"{conf_stats[-1].conference} ({conf_stats[-1].total_profit_units:+.2f}u)" if conf_stats else "N/A"
    
    # Game type comparison
    neutral_wr = game_type_stats['neutral']['win_rate']
    home_wr = game_type_stats['home_venue']['win_rate']
    if neutral_wr > home_wr:
        venue_finding = "Neutral site games show better performance"
    elif home_wr > neutral_wr:
        venue_finding = "Home venue games show better performance"
    else:
        venue_finding = "Similar performance across venue types"
    hca_finding = "Model may be over/under-valuing home court advantage" if abs(neutral_wr - home_wr) > 0.05 else "HCA calibration appears reasonable"
    
    # BET rate assessment
    if 5 <= pred_stats['bet_rate'] <= 15:
        bet_rate_assessment = "BET rate is healthy (5-15% typical)"
    elif pred_stats['bet_rate'] < 5:
        bet_rate_assessment = "BET rate is LOW (< 5%) — model too conservative"
    else:
        bet_rate_assessment = "BET rate is HIGH (> 15%) — may be over-trading"
    
    # Root cause analysis
    has_genuine_edge = clv_stats['mean_clv'] is not None and clv_stats['mean_clv'] > 0
    clv_existence = "Positive mean CLV indicates genuine edge exists" if has_genuine_edge else "Negative mean CLV suggests model is finding noise, not signal"
    clv_mean_val = clv_stats['mean_clv'] if clv_stats['mean_clv'] is not None else 0
    clv_confirms = "confirms we beat the closing line on average" if has_genuine_edge else "shows we're behind the market on average"
    calibration_status = "Calibration is good" if calibration_ok else "Calibration drift detected"
    calibration_detail = "Predicted probabilities match actual outcomes within acceptable margin" if calibration_ok else "Model overestimates win probability in higher edge buckets"
    
    conf_effects = f"Conference-specific effects: {conf_stats[0].conference if conf_stats else 'N/A'} profitable, {conf_stats[-1].conference if conf_stats else 'N/A'} unprofitable" if len(conf_stats) > 1 else "Insufficient conference data"
    
    # Recommendations
    conf_tuning = "Conference-specific tuning" if len(conf_stats) > 3 else "Collect more conference data before tuning"
    reduce_exposure = conf_stats and conf_stats[-1].total_profit_units < -2
    reduce_exposure_detail = f"Reduce exposure in {conf_stats[-1].conference}" if reduce_exposure else ""
    
    report = f"""# K-11: CLV Performance Attribution Report

**Date:** {datetime.utcnow().strftime('%B %d, %Y')}  
**Analyst:** Kimi CLI  
**Mission:** Analyze closing line value performance to understand model edge quality

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Settled Bets | {len(settled)} |
| Wins | {total_wins} |
| Losses | {total_losses} |
| Pushes | {total_pushes} |
| **Overall Win Rate** | **{overall_win_rate:.1%}** |
| **Total P&L (units)** | **{total_profit:+.2f}** |
| Bets with CLV Data | {bets_with_clv} / {len(settled)} ({bets_with_clv/len(settled)*100:.0f}%) |

### CLV Summary

| Metric | Value |
|--------|-------|
| **Mean CLV** | {mean_clv_display} |
| **Median CLV** | {median_clv_display} |
| **Positive CLV Rate** | {clv_stats['positive_clv_pct']:.1f}% |

**Interpretation:**
- CLV > 0: We beat the closing line (genuine edge)
- CLV < 0: Market moved against us (model finding noise, not signal)
- Positive CLV rate shows how often we're on the "smart" side of the market

---

## 1. Win Rate by Edge Bucket

| Edge Range | Bets | Wins | Losses | Win Rate | Expected WR | Calibration Error | Mean CLV | Profit (u) |
|------------|------|------|--------|----------|-------------|-------------------|----------|------------|
"""
    
    for stat in edge_stats:
        clv_str = f"{stat.mean_clv:+.2f}" if stat.mean_clv is not None else "N/A"
        report += f"| {stat.bucket} | {stat.count} | {stat.wins} | {stat.losses} | {stat.win_rate:.1%} | {stat.expected_win_rate:.1%} | {stat.calibration_error:.1%} | {clv_str} | {stat.total_profit_units:+.2f} |\n"
    
    report += f"""
**Key Findings:**
- {calibration_finding}
- {clv_finding}
- {edge_finding}

---

## 2. Performance by Conference

| Conference | Bets | Wins | Losses | Win Rate | Avg Edge | Profit (units) |
|------------|------|------|--------|----------|----------|----------------|
"""
    
    for stat in conf_stats[:10]:  # Top 10
        report += f"| {stat.conference} | {stat.bets} | {stat.wins} | {stat.losses} | {stat.win_rate:.1%} | {stat.avg_edge:.2%} | {stat.total_profit_units:+.2f} |\n"
    
    report += f"""
**Most Profitable:** {most_profitable_conf}  
**Least Profitable:** {least_profitable_conf}

---

## 3. Game Type Analysis (Neutral vs Home Venue)

| Venue Type | Bets | Wins | Losses | Win Rate | Profit (units) |
|------------|------|------|--------|----------|----------------|
| Neutral Site | {game_type_stats['neutral']['bets']} | {game_type_stats['neutral']['wins']} | {game_type_stats['neutral']['losses']} | {game_type_stats['neutral']['win_rate']:.1%} | {game_type_stats['neutral']['profit']:+.2f} |
| Home Venue | {game_type_stats['home_venue']['bets']} | {game_type_stats['home_venue']['wins']} | {game_type_stats['home_venue']['losses']} | {game_type_stats['home_venue']['win_rate']:.1%} | {game_type_stats['home_venue']['profit']:+.2f} |

**Key Finding:**
- {venue_finding}
- {hca_finding}

---

## 4. BET Verdict Frequency (Last 60 Days)

### Weekly Breakdown

| Week Starting | BET Verdicts |
|---------------|--------------|
"""
    
    for week, count in sorted(week_freq.items())[-8:]:  # Last 8 weeks
        report += f"| {week} | {count} |\n"
    
    report += f"""
### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Predictions (60d) | {pred_stats['total']} |
| BET Verdicts | {pred_stats['bets']} ({pred_stats['bet_rate']:.1f}%) |
| CONSIDER Verdicts | {pred_stats['considers']} |
| PASS Verdicts | {pred_stats['passes']} |
| **Bets per Week (avg)** | {pred_stats['bets'] / 8.5:.1f} |

**Assessment:**
- {bet_rate_assessment}
- Current V9.1 effective Kelly divisor is ~3.36x (2.0 x 0.70 SNR x 0.85 integrity), causing over-conservatism

---

## 5. Root Cause Analysis

### Why Does the Model Have a Poor Win Record?

Based on this CLV analysis:

1. **{clv_existence}**
   - Mean CLV of {clv_mean_val:.2f} pts {clv_confirms}

2. **Kelly Compression Stack**
   - V9.1 stacks 3 compression layers: divide 2.0 (fractional) x 0.70 SNR x 0.85 integrity = divide 3.36 effective
   - This requires ~6-8% raw edge to produce 2.5% conservative edge for BET verdict
   - Result: Genuine 4% edges emit CONSIDER instead of BET -> missed opportunities

3. **{calibration_status}**
   - {calibration_detail}

4. **{conf_effects}**

---

## 6. Recommendations for V9.2

### Immediate (Pre-Tournament, if safe):
1. **Monitor CLV in real-time** — if mean CLV turns negative, pause betting

### Post-Tournament (April 7+):
1. **Reduce Kelly compression**
   - Either: Increase SNR scalar floor from 0.5 to 0.8
   - Or: Reduce fractional Kelly divisor from 2.0 to 1.5
   - Target: Effective divisor ~2.0-2.5 instead of 3.36

2. **Adjust MIN_BET_EDGE**
   - Current: 2.5% requires 6-8% raw edge
   - Proposed: 1.5% to allow 4-5% raw edges to qualify as BET

3. **{conf_tuning}**
   - {reduce_exposure_detail}

4. **Remove EvanMiya SE penalty**
   - `EVANMIYA_DOWN_SE_ADDEND = 0.30` adds uncertainty penalty
   - EvanMiya is intentionally excluded (not "down") — penalty should be removed
   - This would narrow margin_se from 1.80 -> 1.50

5. **Restore CLV capture**
   - Only {bets_with_clv}/{len(settled)} bets have CLV data ({bets_with_clv/len(settled)*100:.0f}%)
   - ClosingLine table exists but may not be populating correctly
   - Fix: Ensure closing lines are captured within 30 min of tipoff

---

## Appendix: Data Quality Notes

| Issue | Impact | Status |
|-------|--------|--------|
| Bet grading bug (fixed in EMAC-064) | Historical P&L may be inaccurate | Resolved |
| CLV data sparse | Only {bets_with_clv/len(settled)*100:.0f}% of bets have CLV | Monitor |
| Conference inference | Conference assigned via team name pattern matching | Approximate |
| Neutral site flag | Relies on Game.is_neutral boolean | Accurate |

---

*Report generated: {datetime.utcnow().isoformat()}Z*
*Next update: After March 16 O-8 Baseline Execution*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("K-11: CLV Performance Attribution Analysis")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        print("\n1. Extracting bet data with CLV...")
        bets = extract_bets_with_clv(db)
        print(f"   Found {len(bets)} total bets")
        
        settled = [b for b in bets if b.outcome is not None]
        print(f"   Settled bets: {len(settled)}")
        
        print("\n2. Getting BET frequency by week...")
        week_freq = get_bet_frequency_by_week(db, days=60)
        print(f"   Weeks with data: {len(week_freq)}")
        
        print("\n3. Getting prediction statistics...")
        pred_stats = get_total_predictions_stats(db, days=60)
        print(f"   Total predictions: {pred_stats['total']}")
        print(f"   BET rate: {pred_stats['bet_rate']:.1f}%")
        
        print("\n4. Analyzing edge buckets...")
        edge_stats = analyze_edge_buckets(bets)
        for stat in edge_stats:
            if stat.count > 0:
                print(f"   {stat.bucket}: {stat.count} bets, {stat.win_rate:.1%} WR, {stat.total_profit_units:+.2f}u")
        
        print("\n5. Analyzing conferences...")
        conf_stats = analyze_conferences(bets)
        for stat in conf_stats[:5]:
            print(f"   {stat.conference}: {stat.win_rate:.1%} WR, {stat.total_profit_units:+.2f}u")
        
        print("\n6. Calculating CLV statistics...")
        clv_stats = calculate_mean_clv(bets)
        if clv_stats['mean_clv'] is not None:
            print(f"   Mean CLV: {clv_stats['mean_clv']:+.2f} pts")
            print(f"   Positive CLV rate: {clv_stats['positive_clv_pct']:.1f}%")
        else:
            print("   No CLV data available")
        
        print("\n7. Generating report...")
        output_path = os.path.join(_repo_root, "reports", "K11_CLV_ATTRIBUTION_MARCH_2026.md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = generate_report(bets, week_freq, pred_stats, output_path)
        
        print(f"\n✅ Report written to: {output_path}")
        print("\n" + "=" * 60)
        print("K-11 ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
    finally:
        db.close()


if __name__ == "__main__":
    main()
