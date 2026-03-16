"""
Sharp Money Detection — P1

Detects sharp money/steam signals from line movement patterns:
1. Steam detection: ≥1.5 pts rapid shift in <30 min across multiple books
2. Opener gap: current line vs. open line divergence magnitude
3. Reverse Line Movement (RLM): line moves against public betting % (requires public data)

Integration: Stored in Prediction.full_analysis["sharp_signal"] field
Usage: analysis.py checks signals and adjusts edge confidence
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)


class SharpPattern(Enum):
    REVERSE_LINE_MOVEMENT = "reverse_line_movement"
    STEAM = "steam"
    OPENER_GAP = "opener_gap"
    NONE = "none"


@dataclass
class SharpSignal:
    """Sharp money signal for a game."""
    game_key: str
    side: Optional[str]  # "home", "away", or None
    confidence: float  # 0.0-1.0
    pattern: SharpPattern
    details: Dict[str, Any]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def aligns_with(self, model_side: str) -> bool:
        """Check if sharp signal aligns with model prediction."""
        if self.side is None:
            return False
        return self.side == model_side
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON storage."""
        return {
            "game_key": self.game_key,
            "side": self.side,
            "confidence": round(self.confidence, 3),
            "pattern": self.pattern.value,
            "details": self.details,
            "timestamp": self.timestamp,
            "detected": self.pattern != SharpPattern.NONE,
        }
    
    @classmethod
    def none(cls, game_key: str, reason: str = "", details: Optional[Dict[str, Any]] = None) -> "SharpSignal":
        """Create a 'no signal' instance."""
        return cls(
            game_key=game_key,
            side=None,
            confidence=0.0,
            pattern=SharpPattern.NONE,
            details=details if details is not None else {"reason": reason or "no_pattern_detected"},
        )


class SharpMoneyDetector:
    """
    Detect sharp money patterns from line snapshots.
    
    Designed to work with existing odds_monitor.py infrastructure.
    Pass line history from any source (DB, in-memory, API).
    """
    
    def __init__(self):
        self.steam_threshold_pts = get_float_env("STEAM_THRESHOLD_PTS", "1.5")
        self.steam_window_minutes = get_float_env("STEAM_WINDOW_MINUTES", "30")
        self.opener_gap_threshold = get_float_env("OPENER_GAP_THRESHOLD", "2.0")
        self.rlm_public_threshold = get_float_env("RLM_PUBLIC_THRESHOLD", "60")
        
    def detect_from_history(
        self,
        game_key: str,
        line_history: List[Dict[str, Any]],
        current_home_spread: float,
    ) -> SharpSignal:
        """
        Analyze sharp patterns from line history.
        
        Args:
            game_key: Game identifier (e.g., "Duke@UNC")
            line_history: List of dicts with keys:
                - timestamp (ISO string or datetime)
                - home_spread (float)
            current_home_spread: Current consensus home spread
            
        Returns:
            SharpSignal with highest confidence pattern
        """
        if not line_history:
            return SharpSignal.none(game_key, "insufficient_history")
        
        # Sort by timestamp
        sorted_history = sorted(
            line_history,
            key=lambda x: x.get("timestamp", datetime.utcnow())
        )
        
        # Check each pattern type
        signals = []
        
        # 1. Steam detection
        steam = self._detect_steam(sorted_history, current_home_spread)
        if steam:
            signals.append(steam)
        
        # 2. Opener gap
        opener = self._detect_opener_gap(sorted_history, current_home_spread)
        if opener:
            signals.append(opener)
        
        if not signals:
            return SharpSignal.none(
                game_key,
                details={
                    "history_points": len(sorted_history),
                    "first_line": sorted_history[0].get("home_spread"),
                    "current_line": current_home_spread,
                }
            )
        
        # Return highest confidence signal
        best = max(signals, key=lambda s: s.confidence)
        best.game_key = game_key
        return best
    
    def _detect_steam(
        self,
        history: List[Dict[str, Any]],
        current_home_spread: float,
    ) -> Optional[SharpSignal]:
        """
        Detect steam: rapid line movement ≥1.5 pts in <30 minutes.
        """
        if len(history) < 2:
            return None
        
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.steam_window_minutes)
        
        # Filter to recent history within window
        recent = []
        for h in history:
            ts = h.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    continue
            if ts and ts >= window_start:
                spread = h.get("home_spread")
                if spread is not None:
                    recent.append((ts, spread))
        
        if len(recent) < 2:
            return None
        
        spreads = [s for _, s in recent]
        spread_range = max(spreads) - min(spreads)
        
        if spread_range >= self.steam_threshold_pts:
            first_spread = spreads[0]
            last_spread = spreads[-1]
            movement = last_spread - first_spread
            
            # Steam toward home if line moves more negative
            # Steam toward away if line moves more positive
            if movement < 0:
                side = "home"
                confidence = min(0.9, abs(movement) / 3.0)
            else:
                side = "away"
                confidence = min(0.9, abs(movement) / 3.0)
            
            return SharpSignal(
                game_key="",  # Filled by caller
                side=side,
                confidence=confidence,
                pattern=SharpPattern.STEAM,
                details={
                    "movement_pts": round(movement, 2),
                    "window_minutes": self.steam_window_minutes,
                    "data_points": len(recent),
                    "spread_range": round(spread_range, 2),
                }
            )
        
        return None
    
    def _detect_opener_gap(
        self,
        history: List[Dict[str, Any]],
        current_home_spread: float,
    ) -> Optional[SharpSignal]:
        """
        Detect significant gap between opener and current line.
        """
        if not history:
            return None
        
        opener = history[0].get("home_spread")
        if opener is None:
            return None
        
        gap = abs(current_home_spread - opener)
        
        if gap >= self.opener_gap_threshold:
            if current_home_spread < opener:
                side = "home"
            else:
                side = "away"
            
            confidence = min(0.85, gap / 4.0)
            
            # Get opener timestamp
            opener_ts = history[0].get("timestamp", datetime.utcnow())
            if isinstance(opener_ts, str):
                try:
                    opener_ts = datetime.fromisoformat(opener_ts.replace('Z', '+00:00'))
                except:
                    opener_ts = datetime.utcnow()
            
            hours_since = (datetime.utcnow() - opener_ts).total_seconds() / 3600
            
            return SharpSignal(
                game_key="",  # Filled by caller
                side=side,
                confidence=confidence,
                pattern=SharpPattern.OPENER_GAP,
                details={
                    "opener": opener,
                    "current": current_home_spread,
                    "gap_pts": round(gap, 2),
                    "hours_since_open": round(hours_since, 1),
                }
            )
        
        return None
    
    def detect_rlm(
        self,
        game_key: str,
        line_history: List[Dict[str, Any]],
        current_home_spread: float,
        public_home_pct: float,
    ) -> SharpSignal:
        """
        Detect Reverse Line Movement: line moves against public betting %.
        
        Args:
            game_key: Game identifier
            line_history: Line history
            current_home_spread: Current home spread
            public_home_pct: Percentage of public bets on home (e.g., 65 for 65%)
            
        Returns:
            SharpSignal (may be NONE if no RLM detected)
        """
        if not line_history:
            return SharpSignal.none(game_key, "insufficient_history")
        
        opener = line_history[0].get("home_spread")
        if opener is None:
            return SharpSignal.none(game_key, "no_opener")
        
        movement = current_home_spread - opener
        
        is_heavy_home_public = public_home_pct >= self.rlm_public_threshold
        is_heavy_away_public = public_home_pct <= (100 - self.rlm_public_threshold)
        
        rlm_detected = False
        side = None
        
        if is_heavy_home_public and movement > 0.5:
            # Public on home, line moving toward away = RLM
            rlm_detected = True
            side = "away"
        elif is_heavy_away_public and movement < -0.5:
            # Public on away, line moving toward home = RLM
            rlm_detected = True
            side = "home"
        
        if rlm_detected:
            confidence = min(0.95, abs(movement) / 2.5)
            return SharpSignal(
                game_key=game_key,
                side=side,
                confidence=confidence,
                pattern=SharpPattern.REVERSE_LINE_MOVEMENT,
                details={
                    "public_home_pct": public_home_pct,
                    "movement_pts": round(movement, 2),
                    "opener": opener,
                    "current": current_home_spread,
                }
            )
        
        return SharpSignal.none(game_key, "no_rlm_detected")


# Singleton instance for convenience
_detector = None

def get_detector() -> SharpMoneyDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = SharpMoneyDetector()
    return _detector


def detect_sharp_signal(
    game_key: str,
    line_history: List[Dict[str, Any]],
    current_home_spread: float,
) -> SharpSignal:
    """
    Convenience function to detect sharp signal from line history.
    
    Args:
        game_key: Game identifier
        line_history: List of line snapshots
        current_home_spread: Current consensus home spread
        
    Returns:
        SharpSignal with detected pattern
    """
    detector = get_detector()
    return detector.detect_from_history(game_key, line_history, current_home_spread)


def apply_sharp_adjustment(
    base_edge: float,
    signal: SharpSignal,
    model_side: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Adjust model edge based on sharp signal alignment.
    
    Args:
        base_edge: Original model edge (as decimal, e.g., 0.035 for 3.5%)
        signal: SharpSignal from detection
        model_side: Side model favors ("home" or "away")
        
    Returns:
        Tuple of (adjusted_edge, adjustment_details_dict)
    """
    if signal.pattern == SharpPattern.NONE:
        return base_edge, {"sharp_adjusted": False, "reason": "no_signal"}
    
    aligned = signal.aligns_with(model_side)
    
    # Adjustment magnitudes
    ALIGNMENT_BOOST = 0.005  # +0.5% edge when sharp agrees
    OPPOSITION_PENALTY = -0.008  # -0.8% edge when sharp disagrees
    
    if aligned:
        adjustment = ALIGNMENT_BOOST * signal.confidence
        action = "boost"
    else:
        adjustment = OPPOSITION_PENALTY * signal.confidence
        action = "reduce"
    
    adjusted_edge = base_edge + adjustment
    
    # Ensure edge doesn't go negative
    adjusted_edge = max(0.0, adjusted_edge)
    
    details = {
        "sharp_adjusted": True,
        "action": action,
        "aligned": aligned,
        "sharp_side": signal.side,
        "model_side": model_side,
        "confidence": round(signal.confidence, 3),
        "pattern": signal.pattern.value,
        "base_edge": round(base_edge, 4),
        "adjustment": round(adjustment, 4),
        "adjusted_edge": round(adjusted_edge, 4),
    }
    
    return adjusted_edge, details
