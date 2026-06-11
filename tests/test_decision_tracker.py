"""
Tests for DecisionTracker override comparison logic and /api/fantasy/decisions/accuracy.

Wave 5 — P2 BUGFIX: Fix decision tracker override comparison.
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from backend.fantasy_baseball.decision_tracker import (
    DecisionTracker,
    PlayerDecision,
    DailyAccuracy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_decision(
    decision_id: str,
    date: str = "2026-05-20",
    recommended_action: str = "START",
    user_action: str = None,
    outcome: str = "success",
    accuracy_score: float = 0.8,
) -> PlayerDecision:
    return PlayerDecision(
        decision_id=decision_id,
        date=date,
        player_name="Test Player",
        player_id="mlb.p.12345",
        team="NYY",
        recommended_action=recommended_action,
        confidence=80,
        factors=["platoon_advantage"],
        opponent="BOS",
        opposing_pitcher="Sale",
        venue="Yankee Stadium",
        weather_factor=1.05,
        projected_stats={"hr": 0.15, "r": 0.6, "rbi": 0.5},
        user_action=user_action,
        outcome=outcome,
        accuracy_score=accuracy_score,
    )


@pytest.fixture
def tracker_with_file(tmp_path):
    """DecisionTracker pointing at a temp JSONL file."""
    tracker = DecisionTracker()
    tracker.decisions_file = tmp_path / "decisions.jsonl"
    return tracker


def _write_decisions(tracker: DecisionTracker, decisions: list) -> None:
    with open(tracker.decisions_file, "w") as f:
        for d in decisions:
            if isinstance(d, PlayerDecision):
                from dataclasses import asdict
                f.write(json.dumps(asdict(d)) + "\n")
            else:
                f.write(json.dumps(d) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for override comparison logic
# ─────────────────────────────────────────────────────────────────────────────

class TestOverrideComparison:

    def test_override_better_when_system_wrong(self, tracker_with_file):
        """User who overrode a failing system recommendation counts as override_better."""
        decisions = [
            _make_decision("d1", user_action="OVERRIDE", outcome="failure"),  # user better
            _make_decision("d2", user_action=None, outcome="success"),        # followed
        ]
        _write_decisions(tracker_with_file, decisions)

        acc = tracker_with_file.get_daily_accuracy("2026-05-20")
        assert acc is not None
        assert acc.override_better_count == 1
        assert acc.override_worse_count == 0

    def test_override_worse_when_system_right(self, tracker_with_file):
        """User who overrode a correct system recommendation counts as override_worse."""
        decisions = [
            _make_decision("d1", user_action="OVERRIDE", outcome="success"),  # user worse
            _make_decision("d2", user_action=None, outcome="success"),
        ]
        _write_decisions(tracker_with_file, decisions)

        acc = tracker_with_file.get_daily_accuracy("2026-05-20")
        assert acc is not None
        assert acc.override_better_count == 0
        assert acc.override_worse_count == 1

    def test_mixed_overrides(self, tracker_with_file):
        """Both counts populated when user overrides have mixed outcomes."""
        decisions = [
            _make_decision("d1", user_action="OVERRIDE", outcome="failure"),  # better
            _make_decision("d2", user_action="OVERRIDE", outcome="success"),  # worse
            _make_decision("d3", user_action="OVERRIDE", outcome="failure"),  # better
            _make_decision("d4", user_action=None, outcome="success"),        # followed
        ]
        _write_decisions(tracker_with_file, decisions)

        acc = tracker_with_file.get_daily_accuracy("2026-05-20")
        assert acc is not None
        assert acc.override_better_count == 2
        assert acc.override_worse_count == 1

    def test_no_overrides_both_zero(self, tracker_with_file):
        """No overrides → both counts are 0, no division errors."""
        decisions = [
            _make_decision("d1", user_action=None, outcome="success"),
            _make_decision("d2", user_action=None, outcome="failure"),
        ]
        _write_decisions(tracker_with_file, decisions)

        acc = tracker_with_file.get_daily_accuracy("2026-05-20")
        assert acc is not None
        assert acc.override_better_count == 0
        assert acc.override_worse_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                yield client


class TestDecisionsAccuracyEndpoint:

    def _mock_tracker(self, override_better=2, override_worse=1):
        mock = MagicMock()
        mock.get_daily_accuracy.return_value = DailyAccuracy(
            date="2026-05-20",
            total_decisions=10,
            followed_recommendations=7,
            overrides=3,
            correct_predictions=7,
            incorrect_predictions=3,
            high_conf_accuracy=0.85,
            med_conf_accuracy=0.70,
            low_conf_accuracy=0.50,
            override_better_count=override_better,
            override_worse_count=override_worse,
            start_success_rate=0.75,
            bench_success_rate=0.65,
        )
        return mock

    def test_response_structure(self, fantasy_client):
        """Response has all required fields."""
        with patch(
            "backend.fantasy_baseball.decision_tracker.get_decision_tracker",
            return_value=self._mock_tracker(),
        ):
            resp = fantasy_client.get("/api/fantasy/decisions/accuracy")

        assert resp.status_code == 200
        data = resp.json()
        assert "date" in data
        assert "total_overrides" in data
        assert "better_count" in data
        assert "worse_count" in data
        assert "override_accuracy_pct" in data
        assert "daily_trend" in data
        assert isinstance(data["daily_trend"], list)
        assert len(data["daily_trend"]) == 14

    def test_override_accuracy_pct_computed(self, fantasy_client):
        """override_accuracy_pct = better / (better + worse)."""
        with patch(
            "backend.fantasy_baseball.decision_tracker.get_decision_tracker",
            return_value=self._mock_tracker(override_better=3, override_worse=1),
        ):
            resp = fantasy_client.get("/api/fantasy/decisions/accuracy")

        data = resp.json()
        assert data["override_accuracy_pct"] == pytest.approx(0.75, abs=0.01)

    def test_override_accuracy_pct_zero_when_no_overrides(self, fantasy_client):
        """No overrides → override_accuracy_pct is 0.0, not NaN or error."""
        with patch(
            "backend.fantasy_baseball.decision_tracker.get_decision_tracker",
            return_value=self._mock_tracker(override_better=0, override_worse=0),
        ):
            resp = fantasy_client.get("/api/fantasy/decisions/accuracy")

        assert resp.status_code == 200
        assert resp.json()["override_accuracy_pct"] == 0.0

    def test_trend_has_14_points(self, fantasy_client):
        """Trend window always returns exactly 14 data points, oldest first."""
        with patch(
            "backend.fantasy_baseball.decision_tracker.get_decision_tracker",
            return_value=self._mock_tracker(),
        ):
            resp = fantasy_client.get("/api/fantasy/decisions/accuracy")

        trend = resp.json()["daily_trend"]
        assert len(trend) == 14
        # All trend points must have date + accuracy_pct keys
        for point in trend:
            assert "date" in point
            assert "accuracy_pct" in point
