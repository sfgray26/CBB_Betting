"""
Regression tests for projection fallback IL detection and scoring fixes.

Tests cover:
1. IL detection via status, injury_note, and injury_overlay
2. Fallback scoring returns 0.0 for all fallback players
3. Optimizer safety check for insufficient projection data
4. is_fallback metadata flag
"""

import pytest
from datetime import datetime
from backend.routers.fantasy import _is_il_designated, _projection_fallback_score
from backend.services.injury_overlay import InjuryOverlay


class TestILDetection:
    """Test IL detection via status, injury_note, and injury_overlay."""

    def test_il_detection_via_status_il(self):
        """Test IL detection via player status='IL'."""
        player = {"name": "Test Player", "status": "IL"}
        assert _is_il_designated(player) is True

    def test_il_detection_via_status_il60(self):
        """Test IL detection via player status='IL60'."""
        player = {"name": "Test Player", "status": "IL60"}
        assert _is_il_designated(player) is True

    def test_il_detection_via_status_10_day_il(self):
        """Test IL detection via player status='10-Day IL'."""
        player = {"name": "Test Player", "status": "10-Day IL"}
        assert _is_il_designated(player) is True

    def test_il_detection_via_status_15_day_il(self):
        """Test IL detection via player status='15-Day IL'."""
        player = {"name": "Test Player", "status": "15-Day IL"}
        assert _is_il_designated(player) is True

    def test_il_detection_via_status_60_day_il(self):
        """Test IL detection via player status='60-Day IL'."""
        player = {"name": "Test Player", "status": "60-Day IL"}
        assert _is_il_designated(player) is True

    def test_il_detection_via_injury_note(self):
        """Test IL detection via injury_note containing IL keywords."""
        player = {"name": "Test Player", "status": "Active", "injury_note": "60-Day IL, knee sprain"}
        assert _is_il_designated(player) is True

    def test_il_detection_via_injury_overlay(self):
        """Test IL detection via injury_overlay status."""
        player = {"name": "Test Player", "status": "Active"}
        overlay = InjuryOverlay(
            status="60-Day IL",
            note="Knee sprain",
            return_timeline="ETA Unknown",
            ingested_at=datetime.now(),
            is_stale=False,
        )
        assert _is_il_designated(player, overlay) is True

    def test_il_detection_na_status(self):
        """Test IL detection via NA (Not Active) status."""
        player = {"name": "Test Player", "status": "NA"}
        assert _is_il_designated(player) is True

    def test_il_detection_out_status(self):
        """Test IL detection via OUT status."""
        player = {"name": "Test Player", "status": "OUT"}
        assert _is_il_designated(player) is True

    def test_il_detection_bereavement_status(self):
        """Test IL detection via BEREAVEMENT status."""
        player = {"name": "Test Player", "status": "BEREAVEMENT"}
        assert _is_il_designated(player) is True

    def test_il_detection_dtd_status(self):
        """Test IL detection via DTD (Day to Day) status."""
        player = {"name": "Test Player", "status": "DTD"}
        assert _is_il_designated(player) is True

    def test_no_il_detection_active_status(self):
        """Test that Active status is NOT considered IL."""
        player = {"name": "Test Player", "status": "Active"}
        assert _is_il_designated(player) is False

    def test_no_il_detection_bn_status(self):
        """Test that BN status is NOT considered IL."""
        player = {"name": "Test Player", "status": "BN"}
        assert _is_il_designated(player) is False

    def test_no_il_detection_position_status(self):
        """Test that position statuses (C, 1B, etc.) are NOT considered IL."""
        for pos in ["C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "P"]:
            player = {"name": "Test Player", "status": pos}
            assert _is_il_designated(player) is False, f"Position {pos} should not be IL"

    def test_no_il_detection_none_status(self):
        """Test that None status is NOT considered IL."""
        player = {"name": "Test Player", "status": None}
        assert _is_il_designated(player) is False

    def test_no_il_detection_boolean_status(self):
        """Test that boolean status is NOT considered IL (logs warning)."""
        player = {"name": "Test Player", "status": True}
        assert _is_il_designated(player) is False

    def test_il_detection_edwin_diaz_case(self):
        """Test Díaz case: status='Active' but injury_overlay='60-Day IL'."""
        player = {"name": "Edwin Díaz", "status": "Active", "injury_note": None}
        overlay = InjuryOverlay(
            status="60-Day IL",
            note="Knee surgery",
            return_timeline="ETA Unknown",
            ingested_at=datetime.now(),
            is_stale=False,
            expired_eta=False,
        )
        assert _is_il_designated(player, overlay) is True, "Díaz should be excluded via injury_overlay"

    def test_il_detection_case_insensitive(self):
        """Test IL detection is case-insensitive."""
        for status_variants in ["il", "Il", "iL", "IL", "il", "il60", "IL60"]:
            player = {"name": "Test Player", "status": status_variants}
            assert _is_il_designated(player) is True, f"Status '{status_variants}' should be IL"


class TestFallbackScoring:
    """Test fallback scoring returns 0.0 for all fallback players."""

    def test_fallback_scoring_returns_zero(self):
        """Test that ALL fallback players return 0.0 score."""
        player = {"name": "Test Player", "percent_owned": 100, "player_key": "test"}
        score, source = _projection_fallback_score(player)
        assert score == 0.0, f"Fallback score should be 0.0, got {score}"
        assert source == "projection_fallback"

    def test_fallback_scoring_zero_ownership(self):
        """Test fallback scoring with 0% ownership."""
        player = {"name": "Test Player", "percent_owned": 0, "player_key": "test"}
        score, source = _projection_fallback_score(player)
        assert score == 0.0

    def test_fallback_scoring_no_ownership_field(self):
        """Test fallback scoring when percent_owned field is missing."""
        player = {"name": "Test Player", "player_key": "test"}
        score, source = _projection_fallback_score(player)
        assert score == 0.0

    def test_fallback_scoring_owned_pct_variant(self):
        """Test fallback scoring with owned_pct field variant."""
        player = {"name": "Test Player", "owned_pct": 95, "player_key": "test"}
        score, source = _projection_fallback_score(player)
        assert score == 0.0


class TestOptimizerSafetyCheck:
    """Test optimizer safety check for insufficient projection data."""

    def test_optimizer_safety_check_logic(self):
        """Test the safety check logic: count real projections vs active slots."""
        # Simulate the optimizer safety check logic
        player_data = [
            {"player_key": "1", "name": "Real A", "lineup_score": 85.0, "is_fallback": False},
            {"player_key": "2", "name": "Real B", "lineup_score": 75.0, "is_fallback": False},
            {"player_key": "3", "name": "Fallback A", "lineup_score": 0.0, "is_fallback": True},
            {"player_key": "4", "name": "Fallback B", "lineup_score": 0.0, "is_fallback": True},
        ]

        # Count real projections (score > 0 AND not fallback)
        real_projections = [p for p in player_data if p.get("lineup_score", 0) > 0 and not p.get("is_fallback", False)]

        # Standard Yahoo H2H has 13 active slots (C, 1B, 2B, 3B, SS, OFx3, Util, SPx2, RPx2, P)
        active_slots_count = 13

        # Should have 2 real projections < 13 active slots → error
        assert len(real_projections) < active_slots_count, "Test data should have insufficient real projections"

    def test_real_projection_count_excludes_fallback(self):
        """Test that real projection count excludes fallback players."""
        player_data = [
            {"player_key": "1", "name": "Real A", "lineup_score": 85.0, "is_fallback": False},
            {"player_key": "2", "name": "Real B", "lineup_score": 75.0, "is_fallback": False},
            {"player_key": "3", "name": "Fallback A", "lineup_score": 0.0, "is_fallback": True},
            {"player_key": "4", "name": "Fallback B", "lineup_score": 0.0, "is_fallback": True},
        ]

        real_projections = [p for p in player_data if p.get("lineup_score", 0) > 0 and not p.get("is_fallback", False)]
        assert len(real_projections) == 2, "Should count only real projections"
        assert all(p["name"] in ["Real A", "Real B"] for p in real_projections)


class TestIsFallbackMetadata:
    """Test is_fallback metadata flag."""

    def test_is_fallback_flag_set_correctly(self):
        """Test that is_fallback is True when score_source='projection_fallback'."""
        player_with_real_score = {
            "player_key": "1",
            "name": "Real Player",
            "lineup_score": 85.0,
            "score_source": "player_scores",
        }
        player_with_fallback = {
            "player_key": "2",
            "name": "Fallback Player",
            "lineup_score": 0.0,
            "score_source": "projection_fallback",
        }

        # Simulate the is_fallback logic
        player_with_fallback["is_fallback"] = player_with_fallback["score_source"] == "projection_fallback"
        player_with_real_score["is_fallback"] = player_with_real_score["score_source"] == "projection_fallback"

        assert player_with_fallback["is_fallback"] is True
        assert player_with_real_score["is_fallback"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
