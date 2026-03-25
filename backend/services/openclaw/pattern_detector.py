"""
Pattern Detector Agent

Detects systematic vulnerabilities in model performance:

CBB Patterns:
- Conference bias (underperforming in specific conferences)
- Seed-based gaps (1-8 vs 9-16 in tournament)
- HCA issues (home court advantage mispricing)
- Month/day-of-week clustering
- Game total effects (overs/unders by pace)

MLB Patterns:
- Pitch count fatigue (82% confidence, ERA +0.75-1.50)
- Platoon splits (88% confidence, OPS diff >150 pts)
- Coors Field effect (95% confidence, Total +2.5 runs)
- Bullpen overuse, travel fatigue, weather effects

Usage:
    detector = PatternDetector(sport='cbb')
    
    # Analyze a specific game context
    vulns = detector.analyze(game_context)
    
    # Run full pattern sweep
    report = detector.run_sweep(days=30)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple
from enum import Enum
import logging
import statistics
from collections import defaultdict

from .database import OpenClawDB

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Pattern categories for vulnerability detection."""
    # CBB-specific
    CONFERENCE_BIAS = "conference_bias"
    SEED_MISPRICE = "seed_misprice"
    HCA_ERROR = "hca_error"
    MONTH_DRIFT = "month_drift"
    DAY_OF_WEEK_BIAS = "day_of_week_bias"
    TOTAL_PACE_ERROR = "total_pace_error"
    
    # MLB-specific
    PITCH_COUNT_FATIGUE = "pitch_count_fatigue"
    PLATOON_SPLIT = "platoon_split"
    COORS_EFFECT = "coors_effect"
    BULLPEN_OVERUSE = "bullpen_overuse"
    TRAVEL_FATIGUE = "travel_fatigue"
    WEATHER_IMPACT = "weather_impact"
    WIND_WRIGLEY = "wind_wrigley"
    LINEUP_CHURN = "lineup_churn"
    
    # General
    GENERAL_DRIFT = "general_drift"


@dataclass
class Vulnerability:
    """Detected model vulnerability."""
    pattern_type: PatternType
    sport: str
    confidence: float  # 0.0-1.0
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    description: str
    affected_games: int
    edge_impact: float  # projected edge reduction
    recommended_action: str
    sample_win_rate: Optional[float] = None
    expected_win_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_discord_embed(self) -> Dict[str, Any]:
        """Convert to Discord alert format."""
        color_map = {
            'CRITICAL': 0xFF0000,
            'WARNING': 0xFFA500,
            'INFO': 0x3498DB
        }
        
        fields = [
            {'name': 'Sport', 'value': self.sport.upper(), 'inline': True},
            {'name': 'Confidence', 'value': f'{self.confidence*100:.0f}%', 'inline': True},
            {'name': 'Severity', 'value': self.severity, 'inline': True},
            {'name': 'Affected Games', 'value': str(self.affected_games), 'inline': True},
            {'name': 'Edge Impact', 'value': f'{self.edge_impact*100:.1f}%', 'inline': True},
        ]
        
        if self.sample_win_rate is not None:
            fields.append({
                'name': 'Sample Win Rate',
                'value': f'{self.sample_win_rate*100:.1f}%',
                'inline': True
            })
        
        return {
            'title': f'🚨 Pattern Detected: {self.pattern_type.value}',
            'description': self.description,
            'color': color_map.get(self.severity, 0x808080),
            'fields': fields,
            'footer': {'text': f'Recommendation: {self.recommended_action}'},
            'timestamp': self.detected_at.isoformat()
        }


@dataclass
class SweepReport:
    """Complete pattern sweep results."""
    sport: str
    vulnerabilities: List[Vulnerability]
    games_analyzed: int
    days_analyzed: int
    patterns_checked: int
    generated_at: datetime = field(default_factory=datetime.now)
    
    def has_critical(self) -> bool:
        """Check if any CRITICAL vulnerabilities found."""
        return any(v.severity == 'CRITICAL' for v in self.vulnerabilities)
    
    def by_severity(self) -> Dict[str, List[Vulnerability]]:
        """Group vulnerabilities by severity."""
        result = defaultdict(list)
        for v in self.vulnerabilities:
            result[v.severity].append(v)
        return dict(result)


class PatternDetector:
    """
    Detects systematic vulnerabilities in model predictions.
    
    Implements pattern detection per OpenClaw spec:
    - CBB: Conference, seed, HCA, month, day-of-week, total clustering
    - MLB: Pitch fatigue, platoon splits, Coors effect, weather, etc.
    
    Usage:
        detector = PatternDetector(sport='cbb')
        
        # Full analysis
        report = detector.run_sweep(days=30)
        
        # Send alerts for critical findings
        for vuln in report.vulnerabilities:
            if vuln.severity in ('CRITICAL', 'WARNING'):
                discord.send_alert(vuln.to_discord_embed())
    """
    
    # Statistical thresholds
    MIN_SAMPLE_SIZE = 15
    MIN_CONFIDENCE = 0.70
    SIGNIFICANCE_THRESHOLD = 0.05  # p-value
    
    # Win rate thresholds for flagging
    UNDERPERFORM_THRESHOLD = 0.45  # < 45% win rate
    OVERPERFORM_THRESHOLD = 0.65   # > 65% win rate (also suspicious)
    
    def __init__(
        self,
        sport: str = 'cbb',
        db: Optional[OpenClawDB] = None
    ):
        self.sport = sport.lower()
        self.db = db or OpenClawDB()
        
        # Sport-specific configuration
        self._conferences: Set[str] = set()
        self._pattern_weights: Dict[PatternType, float] = {}
        self._configure_sport()
    
    def _configure_sport(self):
        """Configure pattern weights and conferences by sport."""
        if self.sport == 'cbb':
            self._conferences = {
                'ACC', 'Big 12', 'Big East', 'Big Ten', 'SEC',
                'Pac-12', 'AAC', 'A-10', 'MWC', 'WCC'
            }
            self._pattern_weights = {
                PatternType.CONFERENCE_BIAS: 1.0,
                PatternType.SEED_MISPRICE: 0.9,
                PatternType.HCA_ERROR: 0.8,
                PatternType.TOTAL_PACE_ERROR: 0.7,
                PatternType.MONTH_DRIFT: 0.6,
                PatternType.DAY_OF_WEEK_BIAS: 0.5,
            }
        elif self.sport == 'mlb':
            self._pattern_weights = {
                PatternType.PITCH_COUNT_FATIGUE: 1.0,
                PatternType.PLATOON_SPLIT: 0.95,
                PatternType.COORS_EFFECT: 0.95,
                PatternType.BULLPEN_OVERUSE: 0.85,
                PatternType.WIND_WRIGLEY: 0.80,
                PatternType.TRAVEL_FATIGUE: 0.75,
                PatternType.WEATHER_IMPACT: 0.70,
                PatternType.LINEUP_CHURN: 0.65,
            }
    
    def run_sweep(self, days: int = 30) -> SweepReport:
        """
        Run full pattern detection sweep.
        
        Args:
            days: Lookback period for analysis
            
        Returns:
            SweepReport with all detected vulnerabilities
        """
        logger.info(f"Starting pattern sweep for {self.sport}: {days} days")
        
        vulnerabilities = []
        patterns_checked = 0
        
        # Fetch base game data
        games = self.db.get_game_outcomes(self.sport, days=days)
        
        if len(games) < self.MIN_SAMPLE_SIZE:
            logger.warning(f"Insufficient games for sweep: {len(games)}")
            return SweepReport(
                sport=self.sport,
                vulnerabilities=[],
                games_analyzed=len(games),
                days_analyzed=days,
                patterns_checked=0
            )
        
        # Run sport-specific pattern checks
        if self.sport == 'cbb':
            vulns, checked = self._check_cbb_patterns(games)
        elif self.sport == 'mlb':
            vulns, checked = self._check_mlb_patterns(games)
        else:
            vulns, checked = [], 0
        
        vulnerabilities.extend(vulns)
        patterns_checked += checked
        
        # General drift check (all sports)
        drift_vuln = self._check_general_drift(games)
        if drift_vuln:
            vulnerabilities.append(drift_vuln)
        patterns_checked += 1
        
        report = SweepReport(
            sport=self.sport,
            vulnerabilities=vulnerabilities,
            games_analyzed=len(games),
            days_analyzed=days,
            patterns_checked=patterns_checked
        )
        
        # Log summary
        by_sev = report.by_severity()
        logger.info(
            f"Pattern sweep complete: {len(vulnerabilities)} found "
            f"({len(by_sev.get('CRITICAL', []))} critical, "
            f"{len(by_sev.get('WARNING', []))} warning)"
        )
        
        return report
    
    def _check_cbb_patterns(
        self,
        games: List[Dict[str, Any]]
    ) -> Tuple[List[Vulnerability], int]:
        """Run CBB-specific pattern checks."""
        vulnerabilities = []
        patterns_checked = 0
        
        # 1. Conference bias
        for conf in self._conferences:
            vuln = self._analyze_conference_performance(games, conf)
            if vuln:
                vulnerabilities.append(vuln)
            patterns_checked += 1
        
        # 2. Seed mispricing (tournament only)
        vuln = self._analyze_seed_misprice(games)
        if vuln:
            vulnerabilities.append(vuln)
        patterns_checked += 1
        
        # 3. HCA issues
        vuln = self._analyze_hca_error(games)
        if vuln:
            vulnerabilities.append(vuln)
        patterns_checked += 1
        
        # 4. Month drift
        vuln = self._analyze_month_drift(games)
        if vuln:
            vulnerabilities.append(vuln)
        patterns_checked += 1
        
        # 5. Day of week bias
        vuln = self._analyze_day_of_week(games)
        if vuln:
            vulnerabilities.append(vuln)
        patterns_checked += 1
        
        return vulnerabilities, patterns_checked
    
    def _check_mlb_patterns(
        self,
        games: List[Dict[str, Any]]
    ) -> Tuple[List[Vulnerability], int]:
        """Run MLB-specific pattern checks."""
        vulnerabilities = []
        patterns_checked = 0
        
        # Note: MLB patterns require richer data (pitch counts, weather, etc.)
        # This is a framework - actual detection requires additional data sources
        
        # 1. Pitch count fatigue (would need pitch_count in game data)
        vuln = self._check_pitch_fatigue(games)
        if vuln:
            vulnerabilities.append(vuln)
        patterns_checked += 1
        
        # 2. Platoon splits (would need batter/pitcher handedness)
        vuln = self._check_platoon_splits(games)
        if vuln:
            vulnerabilities.append(vuln)
        patterns_checked += 1
        
        # 3. Coors Field effect
        vuln = self._check_coors_effect(games)
        if vuln:
            vulnerabilities.append(vuln)
        patterns_checked += 1
        
        return vulnerabilities, patterns_checked
    
    def _analyze_conference_performance(
        self,
        games: List[Dict[str, Any]],
        conference: str
    ) -> Optional[Vulnerability]:
        """Detect underperformance in specific conference."""
        conf_games = [g for g in games if g.get('conference') == conference]
        
        if len(conf_games) < self.MIN_SAMPLE_SIZE:
            return None
        
        wins = sum(1 for g in conf_games if self._is_win(g))
        win_rate = wins / len(conf_games)
        
        # Flag if significantly underperforming
        if win_rate < self.UNDERPERFORM_THRESHOLD:
            confidence = min(1.0, len(conf_games) / 50) * self._pattern_weights[PatternType.CONFERENCE_BIAS]
            
            return Vulnerability(
                pattern_type=PatternType.CONFERENCE_BIAS,
                sport=self.sport,
                confidence=confidence,
                severity='WARNING' if confidence > 0.8 else 'INFO',
                description=f"Model underperforming in {conference}: {win_rate*100:.1f}% win rate",
                affected_games=len(conf_games),
                sample_win_rate=win_rate,
                expected_win_rate=0.52,  # Assumed model edge
                edge_impact=0.52 - win_rate,
                recommended_action=f"Review {conference} team ratings and adjust for conference-specific pace/style",
                metadata={'conference': conference}
            )
        
        return None
    
    def _analyze_seed_misprice(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """Detect seed-based mispricing in tournament games."""
        # Filter to games with seed data (tournament)
        tourney_games = [g for g in games if g.get('home_seed') and g.get('away_seed')]
        
        if len(tourney_games) < self.MIN_SAMPLE_SIZE:
            return None
        
        # Compare high seeds (1-8) vs low seeds (9-16)
        high_seed_games = [g for g in tourney_games 
                          if g['home_seed'] <= 8 and g['away_seed'] <= 8]
        low_seed_games = [g for g in tourney_games 
                         if g['home_seed'] >= 9 or g['away_seed'] >= 9]
        
        if not high_seed_games or not low_seed_games:
            return None
        
        high_wins = sum(1 for g in high_seed_games if self._is_win(g))
        low_wins = sum(1 for g in low_seed_games if self._is_win(g))
        
        high_wr = high_wins / len(high_seed_games) if high_seed_games else 0
        low_wr = low_wins / len(low_seed_games) if low_seed_games else 0
        
        # Flag significant disparity
        if abs(high_wr - low_wr) > 0.15:
            confidence = min(1.0, len(tourney_games) / 40) * self._pattern_weights[PatternType.SEED_MISPRICE]
            
            worse = "high-seed" if high_wr < low_wr else "low-seed"
            
            return Vulnerability(
                pattern_type=PatternType.SEED_MISPRICE,
                sport=self.sport,
                confidence=confidence,
                severity='WARNING',
                description=f"Seed-based mispricing: {worse} games performing poorly "
                          f"({high_wr*100:.0f}% vs {low_wr*100:.0f}%)",
                affected_games=len(tourney_games),
                sample_win_rate=min(high_wr, low_wr),
                expected_win_rate=0.52,
                edge_impact=abs(high_wr - low_wr) / 2,
                recommended_action="Review seed-based priors and upset probability model",
                metadata={'high_seed_wr': high_wr, 'low_seed_wr': low_wr}
            )
        
        return None
    
    def _analyze_hca_error(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """Detect home court advantage mispricing."""
        # Separate home/away/neutral games
        home_games = [g for g in games if not g.get('is_neutral_site')]
        neutral_games = [g for g in games if g.get('is_neutral_site')]
        
        if len(home_games) < self.MIN_SAMPLE_SIZE or len(neutral_games) < 10:
            return None
        
        home_wins = sum(1 for g in home_games if self._is_win(g))
        neutral_wins = sum(1 for g in neutral_games if self._is_win(g))
        
        home_wr = home_wins / len(home_games)
        neutral_wr = neutral_wins / len(neutral_games) if neutral_games else 0.5
        
        # Neutral site should perform similarly if HCA priced correctly
        # Large disparity suggests HCA mispricing
        if abs(home_wr - neutral_wr) > 0.10:
            confidence = min(1.0, len(games) / 60) * self._pattern_weights[PatternType.HCA_ERROR]
            
            return Vulnerability(
                pattern_type=PatternType.HCA_ERROR,
                sport=self.sport,
                confidence=confidence,
                severity='WARNING',
                description=f"HCA mispricing suspected: {home_wr*100:.0f}% home vs "
                          f"{neutral_wr*100:.0f}% neutral",
                affected_games=len(games),
                sample_win_rate=neutral_wr,
                expected_win_rate=home_wr,
                edge_impact=abs(home_wr - neutral_wr),
                recommended_action="Recalibrate HCA factor in venue-adjusted ratings",
                metadata={'home_wr': home_wr, 'neutral_wr': neutral_wr}
            )
        
        return None
    
    def _analyze_month_drift(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """Detect performance drift by month (late season fatigue?)."""
        by_month = defaultdict(list)
        for g in games:
            month = g.get('game_date', datetime.now()).month if isinstance(g.get('game_date'), datetime) else 1
            by_month[month].append(g)
        
        if len(by_month) < 2:
            return None
        
        month_rates = {}
        for month, month_games in by_month.items():
            if len(month_games) >= 10:
                wins = sum(1 for g in month_games if self._is_win(g))
                month_rates[month] = wins / len(month_games)
        
        if len(month_rates) < 2:
            return None
        
        # Check for significant month-to-month variance
        rates = list(month_rates.values())
        if len(rates) > 1:
            variance = statistics.variance(rates) if len(rates) > 1 else 0
            if variance > 0.01:  # Significant month-to-month variance
                worst_month = min(month_rates, key=month_rates.get)
                
                return Vulnerability(
                    pattern_type=PatternType.MONTH_DRIFT,
                    sport=self.sport,
                    confidence=0.7,
                    severity='INFO',
                    description=f"Month-to-month variance detected. "
                              f"Worst: {worst_month} ({month_rates[worst_month]*100:.0f}%)",
                    affected_games=sum(len(by_month[m]) for m in month_rates),
                    sample_win_rate=month_rates[worst_month],
                    expected_win_rate=statistics.mean(rates),
                    edge_impact=variance ** 0.5,
                    recommended_action="Review late-season fatigue and tournament prep adjustments",
                    metadata={'month_rates': month_rates}
                )
        
        return None
    
    def _analyze_day_of_week(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """Detect day-of-week bias (e.g., weekend games)."""
        by_dow = defaultdict(list)
        for g in games:
            date = g.get('game_date', datetime.now())
            if isinstance(date, datetime):
                dow = date.strftime('%A')
                by_dow[dow].append(g)
        
        if len(by_dow) < 3:
            return None
        
        dow_rates = {}
        for dow, dow_games in by_dow.items():
            if len(dow_games) >= 10:
                wins = sum(1 for g in dow_games if self._is_win(g))
                dow_rates[dow] = wins / len(dow_games)
        
        if not dow_rates:
            return None
        
        # Check weekend vs weekday
        weekend = ['Saturday', 'Sunday']
        weekend_games = sum(len(by_dow.get(d, [])) for d in weekend)
        weekend_wins = sum(
            sum(1 for g in by_dow.get(d, []) if self._is_win(g))
            for d in weekend
        )
        
        if weekend_games >= self.MIN_SAMPLE_SIZE:
            weekend_wr = weekend_wins / weekend_games
            weekday_wr = statistics.mean([
                r for d, r in dow_rates.items() if d not in weekend
            ]) if any(d not in weekend for d in dow_rates) else 0.5
            
            if abs(weekend_wr - weekday_wr) > 0.08:
                return Vulnerability(
                    pattern_type=PatternType.DAY_OF_WEEK_BIAS,
                    sport=self.sport,
                    confidence=0.65,
                    severity='INFO',
                    description=f"Weekend vs weekday disparity: "
                              f"{weekend_wr*100:.0f}% vs {weekday_wr*100:.0f}%",
                    affected_games=weekend_games,
                    sample_win_rate=weekend_wr,
                    expected_win_rate=weekday_wr,
                    edge_impact=abs(weekend_wr - weekday_wr),
                    recommended_action="Check for travel/rest day adjustments on weekend games",
                    metadata={'weekend_wr': weekend_wr, 'weekday_wr': weekday_wr}
                )
        
        return None
    
    def _check_pitch_fatigue(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """MLB: Detect pitch count fatigue effects (framework only)."""
        # Would need pitch_count data from games
        # Placeholder for when MLB data layer is built
        return None
    
    def _check_platoon_splits(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """MLB: Detect platoon split mispricing (framework only)."""
        # Would need batter/pitcher handedness data
        # Placeholder for when MLB data layer is built
        return None
    
    def _check_coors_effect(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """MLB: Detect Coors Field total mispricing."""
        # Would need venue data
        # Per spec: 95% confidence, Total +2.5 runs effect
        # Placeholder for when MLB data layer is built
        return None
    
    def _check_general_drift(
        self,
        games: List[Dict[str, Any]]
    ) -> Optional[Vulnerability]:
        """Check for overall model drift across all games."""
        if len(games) < self.MIN_SAMPLE_SIZE * 2:
            return None
        
        # Split into first half vs second half
        mid = len(games) // 2
        first_half = games[:mid]
        second_half = games[mid:]
        
        first_wins = sum(1 for g in first_half if self._is_win(g))
        second_wins = sum(1 for g in second_half if self._is_win(g))
        
        first_wr = first_wins / len(first_half) if first_half else 0
        second_wr = second_wins / len(second_half) if second_half else 0
        
        # Flag significant degradation
        if first_wr > 0.5 and second_wr < first_wr - 0.08:
            return Vulnerability(
                pattern_type=PatternType.GENERAL_DRIFT,
                sport=self.sport,
                confidence=min(0.9, len(games) / 100),
                severity='WARNING' if second_wr < 0.45 else 'INFO',
                description=f"General model drift: {first_wr*100:.0f}% → {second_wr*100:.0f}% "
                          f"(recent performance decline)",
                affected_games=len(second_half),
                sample_win_rate=second_wr,
                expected_win_rate=first_wr,
                edge_impact=first_wr - second_wr,
                recommended_action="Full model recalibration recommended",
                metadata={'first_half_wr': first_wr, 'second_half_wr': second_wr}
            )
        
        return None
    
    def _is_win(self, game: Dict[str, Any]) -> bool:
        """Determine if a game outcome was a win for the model."""
        # Simple spread win check - actual implementation would be more nuanced
        predicted = game.get('predicted_spread', 0)
        actual = game.get('spread_result', 0)
        
        if predicted is None or actual is None:
            return False
        
        # Win if prediction direction matches actual result
        return (predicted > 0 and actual > 0) or (predicted < 0 and actual < 0)
    
    def analyze(self, game_context: Dict[str, Any]) -> List[Vulnerability]:
        """
        Analyze a specific game context for vulnerabilities.
        
        Args:
            game_context: Game metadata (conference, seed, venue, etc.)
            
        Returns:
            List of applicable vulnerabilities
        """
        # This would check active vulnerabilities against game context
        # Used for pre-bet filtering
        active = self.db.get_active_vulnerabilities(sport=self.sport)
        
        applicable = []
        for vuln in active:
            # Check if vulnerability applies to this game context
            if self._vulnerability_applies(vuln, game_context):
                applicable.append(vuln)
        
        return applicable
    
    def _vulnerability_applies(
        self,
        vuln: Vulnerability,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a vulnerability applies to specific game context."""
        # Conference check
        if vuln.pattern_type == PatternType.CONFERENCE_BIAS:
            return context.get('conference') == vuln.metadata.get('conference')
        
        # HCA check
        if vuln.pattern_type == PatternType.HCA_ERROR:
            return context.get('is_neutral_site', False)
        
        # Add more context matching as needed
        return False
