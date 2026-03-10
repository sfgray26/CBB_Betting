"""
Tests for sharp_money.py — P1 Sharp Money Detection
"""

import pytest
from datetime import datetime, timedelta

from backend.services.sharp_money import (
    SharpMoneyDetector,
    SharpSignal,
    SharpPattern,
    detect_sharp_signal,
    apply_sharp_adjustment,
)


class TestSteamDetection:
    """Test steam (rapid line movement) detection."""
    
    def test_no_steam_with_small_movement(self):
        """Should not detect steam if movement is below threshold."""
        detector = SharpMoneyDetector()
        detector.steam_threshold_pts = 1.5
        
        history = [
            {"timestamp": datetime.utcnow() - timedelta(minutes=20), "home_spread": -3.0},
            {"timestamp": datetime.utcnow() - timedelta(minutes=10), "home_spread": -3.5},
        ]
        
        result = detector.detect_from_history("Test@Game", history, -3.5)
        
        assert result.pattern == SharpPattern.NONE
    
    def test_detect_steam_home(self):
        """Should detect steam toward home when line moves negative rapidly."""
        detector = SharpMoneyDetector()
        detector.steam_threshold_pts = 1.5
        detector.steam_window_minutes = 30
        
        now = datetime.utcnow()
        history = [
            {"timestamp": now - timedelta(minutes=25), "home_spread": -3.0},
            {"timestamp": now - timedelta(minutes=10), "home_spread": -4.5},
            {"timestamp": now - timedelta(minutes=5), "home_spread": -5.0},
        ]
        
        result = detector.detect_from_history("Test@Game", history, -5.0)
        
        assert result.pattern == SharpPattern.STEAM
        assert result.side == "home"
        assert result.confidence > 0.3
        assert result.details["movement_pts"] < 0  # Negative = toward home
    
    def test_detect_steam_away(self):
        """Should detect steam toward away when line moves positive rapidly."""
        detector = SharpMoneyDetector()
        detector.steam_threshold_pts = 1.5
        
        now = datetime.utcnow()
        history = [
            {"timestamp": now - timedelta(minutes=20), "home_spread": -5.0},
            {"timestamp": now - timedelta(minutes=10), "home_spread": -3.5},
        ]
        
        result = detector.detect_from_history("Test@Game", history, -3.5)
        
        assert result.pattern == SharpPattern.STEAM
        assert result.side == "away"
        assert result.details["movement_pts"] > 0


class TestOpenerGapDetection:
    """Test opener gap detection."""
    
    def test_no_gap_below_threshold(self):
        """Should not detect gap if movement is below threshold."""
        detector = SharpMoneyDetector()
        detector.opener_gap_threshold = 2.0
        
        history = [
            {"timestamp": datetime.utcnow() - timedelta(hours=12), "home_spread": -3.0},
        ]
        
        result = detector.detect_from_history("Test@Game", history, -4.0)  # 1 pt move
        
        assert result.pattern == SharpPattern.NONE
    
    def test_detect_opener_gap_home(self):
        """Should detect gap when line moves significantly toward home."""
        detector = SharpMoneyDetector()
        detector.opener_gap_threshold = 2.0
        
        now = datetime.utcnow()
        history = [
            {"timestamp": now - timedelta(hours=12), "home_spread": -2.0},  # Opener
        ]
        
        result = detector.detect_from_history("Test@Game", history, -5.0)  # 3 pt move
        
        assert result.pattern == SharpPattern.OPENER_GAP
        assert result.side == "home"
        assert result.details["gap_pts"] == 3.0
        assert result.details["opener"] == -2.0
    
    def test_detect_opener_gap_away(self):
        """Should detect gap when line moves significantly toward away."""
        detector = SharpMoneyDetector()
        detector.opener_gap_threshold = 2.0
        
        history = [
            {"timestamp": datetime.utcnow() - timedelta(hours=12), "home_spread": -5.0},
        ]
        
        result = detector.detect_from_history("Test@Game", history, -1.0)  # 4 pt move
        
        assert result.pattern == SharpPattern.OPENER_GAP
        assert result.side == "away"


class TestRLMDetection:
    """Test reverse line movement detection."""
    
    def test_rlm_public_home_line_away(self):
        """Should detect RLM when public on home but line moves toward away."""
        detector = SharpMoneyDetector()
        
        history = [
            {"timestamp": datetime.utcnow() - timedelta(hours=12), "home_spread": -4.0},
        ]
        
        signal = detector.detect_rlm("Test@Game", history, -2.5, public_home_pct=75)
        
        assert signal.pattern == SharpPattern.REVERSE_LINE_MOVEMENT
        assert signal.side == "away"  # Sharp money on away
        assert signal.confidence > 0.5
    
    def test_rlm_public_away_line_home(self):
        """Should detect RLM when public on away but line moves toward home."""
        detector = SharpMoneyDetector()
        
        history = [
            {"timestamp": datetime.utcnow() - timedelta(hours=12), "home_spread": -2.0},
        ]
        
        signal = detector.detect_rlm("Test@Game", history, -4.5, public_home_pct=25)
        
        assert signal.pattern == SharpPattern.REVERSE_LINE_MOVEMENT
        assert signal.side == "home"  # Sharp money on home
    
    def test_no_rlm_when_line_follows_public(self):
        """Should not detect RLM when line follows public money."""
        detector = SharpMoneyDetector()
        
        history = [
            {"timestamp": datetime.utcnow() - timedelta(hours=12), "home_spread": -4.0},
        ]
        
        # Public on home (75%), line moves more negative (toward home)
        signal = detector.detect_rlm("Test@Game", history, -5.5, public_home_pct=75)
        
        assert signal.pattern == SharpPattern.NONE


class TestEdgeAdjustment:
    """Test edge adjustment based on sharp signals."""
    
    def test_boost_when_aligned(self):
        """Should boost edge when sharp signal aligns with model."""
        signal = SharpSignal(
            game_key="Test@Game",
            side="home",
            confidence=0.8,
            pattern=SharpPattern.STEAM,
            details={},
        )
        
        adjusted, details = apply_sharp_adjustment(0.035, signal, "home")
        
        assert adjusted > 0.035  # Edge increased
        assert details["aligned"] is True
        assert details["action"] == "boost"
    
    def test_reduce_when_opposed(self):
        """Should reduce edge when sharp signal opposes model."""
        signal = SharpSignal(
            game_key="Test@Game",
            side="away",
            confidence=0.8,
            pattern=SharpPattern.STEAM,
            details={},
        )
        
        adjusted, details = apply_sharp_adjustment(0.035, signal, "home")
        
        assert adjusted < 0.035  # Edge decreased
        assert details["aligned"] is False
        assert details["action"] == "reduce"
    
    def test_no_change_when_no_signal(self):
        """Should not change edge when no sharp signal."""
        signal = SharpSignal.none("Test@Game")
        
        adjusted, details = apply_sharp_adjustment(0.035, signal, "home")
        
        assert adjusted == 0.035
        assert details["sharp_adjusted"] is False
    
    def test_edge_never_negative(self):
        """Adjusted edge should never go below zero."""
        signal = SharpSignal(
            game_key="Test@Game",
            side="away",
            confidence=1.0,  # Maximum confidence
            pattern=SharpPattern.STEAM,
            details={},
        )
        
        adjusted, details = apply_sharp_adjustment(0.005, signal, "home")
        
        assert adjusted >= 0.0


class TestSignalSerialization:
    """Test signal serialization for JSON storage."""
    
    def test_to_dict(self):
        """Should convert signal to dict for JSON storage."""
        signal = SharpSignal(
            game_key="Duke@UNC",
            side="home",
            confidence=0.75,
            pattern=SharpPattern.STEAM,
            details={"movement_pts": -2.5},
        )
        
        d = signal.to_dict()
        
        assert d["game_key"] == "Duke@UNC"
        assert d["side"] == "home"
        assert d["confidence"] == 0.75
        assert d["pattern"] == "steam"
        assert d["detected"] is True
        assert "timestamp" in d
    
    def test_none_signal_to_dict(self):
        """Should handle NONE pattern correctly."""
        signal = SharpSignal.none("Test@Game", "insufficient_data")
        
        d = signal.to_dict()
        
        assert d["detected"] is False
        assert d["side"] is None
        assert d["confidence"] == 0.0
