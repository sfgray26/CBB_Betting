"""Tests for UI contract shape and validation."""

from datetime import datetime

import pytest
from zoneinfo import ZoneInfo

from backend.contracts import (
    CategoryStatusTag,
    IPPaceFlag,
    ConstraintBudget,
    FreshnessMetadata,
    CategoryStats,
    MatchupScoreboardRow,
    MatchupScoreboardResponse,
    PlayerGameContext,
    CanonicalPlayerRow,
)
from backend.stat_contract import SCORING_CATEGORY_CODES, BATTING_CODES


def test_category_status_tag_values():
    """CategoryStatusTag has exactly 5 values; all values are lowercase strings."""
    values = [tag.value for tag in CategoryStatusTag]
    assert len(values) == 5
    assert all(v.islower() for v in values)
    assert "locked_win" in values
    assert "locked_loss" in values
    assert "bubble" in values
    assert "leaning_win" in values
    assert "leaning_loss" in values


def test_constraint_budget():
    """ConstraintBudget can be created with sample data; frozen (mutation raises)."""
    budget = ConstraintBudget(
        acquisitions_used=3,
        acquisitions_remaining=5,
        acquisition_limit=8,
        acquisition_warning=False,
        il_used=1,
        il_total=3,
        ip_accumulated=15.0,
        ip_minimum=18.0,
        ip_pace=IPPaceFlag.BEHIND,
        as_of=datetime.now(ZoneInfo("America/New_York")),
    )
    assert budget.acquisitions_used == 3
    assert budget.ip_pace == IPPaceFlag.BEHIND

    # Test frozen
    with pytest.raises(Exception):  # FrozenInstanceError
        budget.acquisitions_used = 4


def test_ip_pace_flag():
    """IPPaceFlag has exactly 3 values."""
    values = [flag.value for flag in IPPaceFlag]
    assert len(values) == 3
    assert "behind" in values
    assert "on_track" in values
    assert "ahead" in values


def test_freshness_metadata():
    """FreshnessMetadata can be created with sample data; verify frozen."""
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=datetime.now(ZoneInfo("America/New_York")),
        computed_at=datetime.now(ZoneInfo("America/New_York")),
        staleness_threshold_minutes=60,
        is_stale=False,
    )
    assert freshness.primary_source == "yahoo"
    assert not freshness.is_stale

    # Test frozen
    with pytest.raises(Exception):
        freshness.is_stale = True


def test_category_stats_validator_success():
    """CategoryStats with 18 canonical-code keys succeeds."""
    values = {code: 0.0 for code in SCORING_CATEGORY_CODES}
    stats = CategoryStats(values=values)
    assert len(stats.values) == 18
    assert stats.values["R"] == 0.0


def test_category_stats_validator_missing_key():
    """CategoryStats validator rejects missing key."""
    values = {code: 0.0 for code in SCORING_CATEGORY_CODES if code != "R"}
    with pytest.raises(ValueError, match="Missing scoring categories"):
        CategoryStats(values=values)


def test_category_stats_validator_extra_key():
    """CategoryStats validator rejects extra key."""
    values = {code: 0.0 for code in SCORING_CATEGORY_CODES}
    values["INVALID_KEY"] = 0.0
    with pytest.raises(ValueError, match="Unexpected category keys"):
        CategoryStats(values=values)


def test_category_stats_uses_contract_codes():
    """Validate that the validator references SCORING_CATEGORY_CODES (not hardcoded)."""
    # This test confirms the validator adapts to the loaded contract
    # by checking it accepts exactly what's in SCORING_CATEGORY_CODES
    values = {code: 0.0 for code in SCORING_CATEGORY_CODES}
    stats = CategoryStats(values=values)
    assert set(stats.values.keys()) == SCORING_CATEGORY_CODES


def test_matchup_scoreboard_row():
    """MatchupScoreboardRow can be instantiated with required fields; frozen."""
    row = MatchupScoreboardRow(
        category="HR_B",
        category_label="HR",
        is_lower_better=False,
        is_batting=True,
        my_current=5.0,
        opp_current=3.0,
        current_margin=2.0,
    )
    assert row.category == "HR_B"
    assert row.is_batting is True

    # Test frozen
    with pytest.raises(Exception):
        row.my_current = 6.0


def test_matchup_scoreboard_row_with_optional_fields():
    """MatchupScoreboardRow with optional Phase 2/3 fields."""
    now = datetime.now(ZoneInfo("America/New_York"))
    row = MatchupScoreboardRow(
        category="ERA",
        category_label="ERA",
        is_lower_better=True,
        is_batting=False,
        my_current=3.50,
        opp_current=4.00,
        current_margin=-0.50,
        my_projected_final=3.45,
        opp_projected_final=4.05,
        projected_margin=-0.60,
        status=CategoryStatusTag.LEANING_WIN,
        flip_probability=0.15,
        delta_to_flip="+0.15 ERA",
        games_remaining=2,
        ip_context="6 IP today",
    )
    assert row.status == CategoryStatusTag.LEANING_WIN
    assert row.games_remaining == 2


def test_matchup_scoreboard_response():
    """MatchupScoreboardResponse can be instantiated with 18 rows + budget + freshness."""
    now = datetime.now(ZoneInfo("America/New_York"))
    rows = []
    for code in SCORING_CATEGORY_CODES:
        row = MatchupScoreboardRow(
            category=code,
            category_label=code,
            is_lower_better=(code in {"ERA", "WHIP", "L", "K_B", "HR_P"}),
            is_batting=(code in BATTING_CODES),
            my_current=0.0,
            opp_current=0.0,
            current_margin=0.0,
        )
        rows.append(row)

    budget = ConstraintBudget(
        acquisitions_used=3,
        acquisitions_remaining=5,
        acquisition_limit=8,
        acquisition_warning=False,
        il_used=1,
        il_total=3,
        ip_accumulated=15.0,
        ip_minimum=18.0,
        ip_pace=IPPaceFlag.ON_TRACK,
        as_of=now,
    )

    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=now,
        computed_at=now,
        staleness_threshold_minutes=60,
        is_stale=False,
    )

    response = MatchupScoreboardResponse(
        week=5,
        opponent_name="Opponent Team",
        categories_won=7,
        categories_lost=5,
        categories_tied=6,
        projected_won=8,
        projected_lost=4,
        projected_tied=6,
        overall_win_probability=0.65,
        rows=rows,
        budget=budget,
        freshness=freshness,
    )
    assert len(response.rows) == 18
    assert response.week == 5


def test_canonical_player_row():
    """CanonicalPlayerRow can be created with full data; verify frozen."""
    now = datetime.now(ZoneInfo("America/New_York"))

    season_values = {code: 0.0 for code in SCORING_CATEGORY_CODES}
    season_stats = CategoryStats(values=season_values)

    game_context = PlayerGameContext(
        opponent="BOS",
        home_away="home",
        game_time=now,
        projected_k=8.0,
        projected_era_impact=-0.15,
    )

    player = CanonicalPlayerRow(
        player_name="John Doe",
        team="NYY",
        eligible_positions=["1B", "OF"],
        status="playing",
        game_context=game_context,
        season_stats=season_stats,
        rolling_7d=None,
        rolling_15d=None,
        rolling_30d=None,
        ros_projection=None,
        row_projection=None,
        ownership_pct=75.0,
        injury_status=None,
        injury_return_timeline=None,
        freshness=FreshnessMetadata(
            primary_source="yahoo",
            fetched_at=now,
            computed_at=now,
            staleness_threshold_minutes=60,
            is_stale=False,
        ),
        yahoo_player_key="mlb.p.12345",
        bdl_player_id=12345,
        mlbam_id=54321,
    )
    assert player.player_name == "John Doe"
    assert player.team == "NYY"
    assert player.game_context.opponent == "BOS"

    # Test frozen
    with pytest.raises(Exception):
        player.player_name = "Jane Doe"


def test_player_game_context():
    """PlayerGameContext can be created with sample data; verify frozen."""
    now = datetime.now(ZoneInfo("America/New_York"))

    context = PlayerGameContext(
        opponent="BOS",
        home_away="away",
        game_time=now,
        opposing_sp_name="Chris Sale",
        opposing_sp_handedness="L",
        projected_impact=0.25,
    )
    assert context.home_away == "away"
    assert context.opposing_sp_handedness == "L"

    # Test frozen
    with pytest.raises(Exception):
        context.opponent = "NYY"


def test_all_contracts_frozen():
    """Verify mutation raises error for every contract class."""
    now = datetime.now(ZoneInfo("America/New_York"))

    # CategoryStatusTag is an Enum, so it's immutable by design

    # ConstraintBudget
    budget = ConstraintBudget(
        acquisitions_used=3,
        acquisitions_remaining=5,
        acquisition_limit=8,
        acquisition_warning=False,
        il_used=1,
        il_total=3,
        ip_accumulated=15.0,
        ip_minimum=18.0,
        ip_pace=IPPaceFlag.ON_TRACK,
        as_of=now,
    )
    with pytest.raises(Exception):
        budget.acquisitions_used = 4

    # FreshnessMetadata
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=now,
        computed_at=now,
        staleness_threshold_minutes=60,
        is_stale=False,
    )
    with pytest.raises(Exception):
        freshness.is_stale = True

    # CategoryStats
    values = {code: 0.0 for code in SCORING_CATEGORY_CODES}
    stats = CategoryStats(values=values)
    with pytest.raises(Exception):
        stats.values = {}

    # MatchupScoreboardRow
    row = MatchupScoreboardRow(
        category="HR_B",
        category_label="HR",
        is_lower_better=False,
        is_batting=True,
        my_current=5.0,
        opp_current=3.0,
        current_margin=2.0,
    )
    with pytest.raises(Exception):
        row.my_current = 6.0

    # PlayerGameContext
    context = PlayerGameContext(
        opponent="BOS",
        home_away="home",
        game_time=now,
    )
    with pytest.raises(Exception):
        context.opponent = "NYY"

    # CanonicalPlayerRow
    season_stats = CategoryStats(values={code: 0.0 for code in SCORING_CATEGORY_CODES})
    player = CanonicalPlayerRow(
        player_name="Test",
        team="NYY",
        eligible_positions=["1B"],
        status="playing",
        season_stats=season_stats,
        freshness=FreshnessMetadata(
            primary_source="yahoo",
            fetched_at=now,
            computed_at=now,
            staleness_threshold_minutes=60,
            is_stale=False,
        ),
    )
    with pytest.raises(Exception):
        player.player_name = "Changed"
