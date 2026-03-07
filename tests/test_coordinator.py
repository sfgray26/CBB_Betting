"""Tests for O-9 tiered escalation coordinator."""

import pytest
from unittest.mock import patch

from backend.services.coordinator import escalate_if_needed


class TestEscalateIfNeeded:
    """Test escalation trigger detection."""

    def test_units_trigger_fires(self):
        """recommended_units >= threshold triggers escalation."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=2.0,
            integrity_verdict="CONFIRMED",
            is_neutral=False,
        )
        assert result is True

    def test_units_below_threshold_no_trigger(self):
        """recommended_units below threshold does not trigger."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=1.0,
            integrity_verdict="CONFIRMED",
            is_neutral=False,
        )
        assert result is False

    def test_neutral_site_triggers(self):
        """is_neutral=True triggers escalation (tournament proxy)."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5,
            integrity_verdict=None,
            is_neutral=True,
        )
        assert result is True

    def test_volatile_verdict_triggers(self):
        """VOLATILE integrity_verdict triggers escalation."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5,
            integrity_verdict="VOLATILE — lineup uncertainty",
            is_neutral=False,
        )
        assert result is True

    def test_caution_verdict_no_trigger(self):
        """CAUTION verdict alone does not trigger escalation."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5,
            integrity_verdict="CAUTION",
            is_neutral=False,
        )
        assert result is False

    def test_none_verdict_no_trigger(self):
        """None integrity_verdict does not trigger escalation."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.5,
            integrity_verdict=None,
            is_neutral=False,
        )
        assert result is False

    def test_multiple_triggers_fires(self):
        """Multiple triggers: all fire, still returns True."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=2.5,
            integrity_verdict="VOLATILE",
            is_neutral=True,
        )
        assert result is True

    def test_exactly_at_threshold_triggers(self):
        """recommended_units exactly equal to threshold triggers (>=)."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=1.5,
            integrity_verdict=None,
            is_neutral=False,
        )
        assert result is True

    @patch.dict("os.environ", {"ESCALATION_UNITS_THRESHOLD": "3.0"})
    def test_env_override_threshold(self):
        """ESCALATION_UNITS_THRESHOLD env var overrides default 1.5."""
        # 2.0 units is below custom threshold of 3.0 — should NOT trigger
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=2.0,
            integrity_verdict=None,
            is_neutral=False,
        )
        assert result is False

    def test_no_triggers_returns_false(self):
        """No triggers: returns False cleanly."""
        result = escalate_if_needed(
            game_key="test_game",
            home_team="Duke",
            away_team="UNC",
            recommended_units=0.0,
            integrity_verdict="CONFIRMED",
            is_neutral=False,
        )
        assert result is False
