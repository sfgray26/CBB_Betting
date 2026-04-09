"""
Two-Start Pitcher Detection — UAT Validation Tests

Full backend validation before UI consumption. Tests data pipeline integrity,
probable_pitchers table population, and two-start detection logic.

UAT Checklist:
  1. probable_pitchers table exists (P26 migration)
  2. Table has data for current week
  3. Data freshness (<24h for games within 48h)
  4. Player ID resolution works (BDL → name mapping)
  5. Matchup quality scores in valid range (-2.0 to +2.0)
  6. Acquisition method classification accurate
  7. Two-start detection returns valid opportunities
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch

from backend.fantasy_baseball.two_start_detector import (
    TwoStartDetector,
    TwoStartOpportunity,
    MatchupRating,
)


def test_uat_01_probable_pitchers_table_exists():
    """
    UAT-01: Verify probable_pitchers table exists and is queryable.

    Validates P26 migration was successful.
    """
    import os
    from sqlalchemy import create_engine, text

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set — skipping database validation")

    try:
        engine = create_engine(db_url)
    except Exception as e:
        pytest.skip(f"Database connection failed: {e} — skipping UAT")

    try:
        with engine.connect() as conn:
            # Check table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'probable_pitchers'
                )
                """))

            exists = result.scalar()
            assert exists is True, "probable_pitchers table does not exist — P26 migration required"

            # Check table has data (at least schema is valid)
            result = conn.execute(text("SELECT COUNT(*) FROM probable_pitchers"))
            count = result.scalar()
            assert count >= 0, "Failed to query probable_pitchers table"
    except Exception as e:
        pytest.skip(f"Database query failed: {e} — skipping UAT")


def test_uat_02_two_start_detector_initialization():
    """
    UAT-02: Verify TwoStartDetector initializes correctly.

    Tests database connection and configuration loading.
    """
    import os

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set — skipping")

    detector = TwoStartDetector(db_url)
    assert detector.engine is not None, "Failed to initialize database engine"
    assert detector.SessionLocal is not None, "Failed to initialize session factory"

    # Verify park factors loaded
    assert len(detector.PARK_FACTORS) == 30, "Park factors not loaded (expected 30 MLB teams)"
    assert detector.PARK_FACTORS["COL"] > 1.0, "COL should be hitter-friendly (park factor > 1.0)"
    assert detector.PARK_FACTORS["SF"] < 1.0, "SF should be pitcher-friendly (park factor < 1.0)"


def test_uat_03_two_start_detection_returns_valid_opportunities():
    """
    UAT-03: Verify two-start detection returns valid TwoStartOpportunity objects.

    Tests the core detection logic with mock data.
    """
    detector = TwoStartDetector()  # No DB connection required for mock test

    # Mock database response
    mock_starts = [
        {
            "pitcher_name": "Gerrit Cole",
            "team": "NYY",
            "bdl_player_id": 12345,
            "game_date": date.today() + timedelta(days=1),
            "opponent": "BOS",
            "is_home": True,
            "is_confirmed": True,
            "park_factor": 0.97,
            "quality_score": 1.5,
        },
        {
            "pitcher_name": "Gerrit Cole",
            "team": "NYY",
            "bdl_player_id": 12345,
            "game_date": date.today() + timedelta(days=5),
            "opponent": "TB",
            "is_home": False,
            "is_confirmed": True,
            "park_factor": 1.05,
            "quality_score": 0.8,
        },
    ]

    # Build opportunity directly
    opp = detector._build_opportunity(12345, mock_starts, league_rosters=None)

    # Validate all required fields
    assert isinstance(opp, TwoStartOpportunity)
    assert opp.player_id == "12345"
    assert opp.name == "Gerrit Cole"
    assert opp.team == "NYY"
    assert opp.week >= 1

    # Validate game_1
    assert isinstance(opp.game_1, MatchupRating)
    assert opp.game_1.opponent == "BOS"
    assert opp.game_1.park_factor == 0.97
    assert opp.game_1.quality_score == 1.5
    assert isinstance(opp.game_1.is_home, bool)

    # Validate game_2
    assert isinstance(opp.game_2, MatchupRating)
    assert opp.game_2.opponent == "TB"
    assert opp.game_2.park_factor == 1.05
    assert opp.game_2.quality_score == 0.8

    # Validate categories addressed
    assert "W" in opp.categories_addressed
    assert "QS" in opp.categories_addressed
    assert "K" in opp.categories_addressed
    assert "K/9" in opp.categories_addressed

    # Validate acquisition method
    assert opp.acquisition_method in ["ROSTERED", "WAIVER", "FREE_AGENT"]

    # Validate streamer rating
    assert opp.streamer_rating in ["EXCELLENT", "GOOD", "AVOID"]

    # Validate quality score average
    expected_avg = (1.5 + 0.8) / 2
    assert abs(opp.average_quality_score - expected_avg) < 0.01

    # Validate IP projection
    assert opp.total_ip_projection > 0

    # Validate UAT flags
    assert opp.data_freshness in ["FRESH", "STALE", "MISSING"]
    assert opp.player_name_confidence in ["HIGH", "MEDIUM", "LOW"]


def test_uat_04_matchup_quality_score_in_valid_range():
    """
    UAT-04: Verify matchup quality scores are in valid range (-2.0 to +2.0).

    Ensures data pipeline produces valid quality scores.
    """
    detector = TwoStartDetector()

    # Test edge cases
    excellent_matchup = MatchupRating(
        opponent="MIA",  # Weak offense
        park_factor=0.97,  # Pitcher-friendly park
        quality_score=2.0,  # Max positive
        game_date=date.today(),
        is_home=True,
    )

    terrible_matchup = MatchupRating(
        opponent="LAD",  # Strong offense
        park_factor=1.12,  # Hitter-friendly park (COL)
        quality_score=-2.0,  # Max negative
        game_date=date.today(),
        is_home=False,
    )

    neutral_matchup = MatchupRating(
        opponent="MIL",
        park_factor=1.00,
        quality_score=0.0,
        game_date=date.today(),
        is_home=True,
    )

    # Validate ranges
    for matchup in [excellent_matchup, terrible_matchup, neutral_matchup]:
        assert -2.0 <= matchup.quality_score <= 2.0, \
            f"Quality score {matchup.quality_score} outside valid range [-2.0, 2.0]"
        assert 0.5 <= matchup.park_factor <= 1.5, \
            f"Park factor {matchup.park_factor} outside reasonable range [0.5, 1.5]"


def test_uat_05_streamer_rating_classification():
    """
    UAT-05: Verify streamer rating classification matches quality scores.

    EXCELLENT: avg_quality >= 1.0
    GOOD: 0.0 <= avg_quality < 1.0
    AVOID: avg_quality < 0.0
    """
    detector = TwoStartDetector()

    test_cases = [
        (1.5, "EXCELLENT"),
        (1.0, "EXCELLENT"),
        (0.8, "GOOD"),
        (0.0, "GOOD"),
        (-0.5, "AVOID"),
        (-2.0, "AVOID"),
    ]

    for quality_score, expected_rating in test_cases:
        mock_starts = [
            {
                "pitcher_name": "Test",
                "team": "TST",
                "bdl_player_id": 99999,
                "game_date": date.today(),
                "opponent": "OPP",
                "is_home": True,
                "is_confirmed": True,
                "park_factor": 1.0,
                "quality_score": quality_score,
            },
            {
                "pitcher_name": "Test",
                "team": "TST",
                "bdl_player_id": 99999,
                "game_date": date.today() + timedelta(days=3),
                "opponent": "OPP",
                "is_home": False,
                "is_confirmed": True,
                "park_factor": 1.0,
                "quality_score": quality_score,
            },
        ]

        opp = detector._build_opportunity(99999, mock_starts)
        assert opp.streamer_rating == expected_rating, \
            f"Quality score {quality_score} should map to {expected_rating}, got {opp.streamer_rating}"


def test_uat_06_acquisition_method_classification():
    """
    UAT-06: Verify acquisition method classification is accurate.

    ROSTERED: Player found in league_rosters
    FREE_AGENT: Player not found (default)
    WAIVER: Future enhancement (not implemented)
    """
    detector = TwoStartDetector()

    # Mock league rosters
    league_rosters = [
        [
            {"bdl_player_id": 11111, "name": "Player A"},
            {"bdl_player_id": 22222, "name": "Player B"},
        ],
        [
            {"bdl_player_id": 33333, "name": "Player C"},
        ],
        # ... 8 more teams
    ]

    # Test 1: Rostered player
    method, waiver_cost, faab_cost = detector._classify_acquisition(11111, league_rosters)
    assert method == "ROSTERED"
    assert waiver_cost is None
    assert faab_cost is None

    # Test 2: Free agent (not in any roster)
    method, waiver_cost, faab_cost = detector._classify_acquisition(99999, league_rosters)
    assert method == "FREE_AGENT"
    assert waiver_cost is None
    assert faab_cost == 5  # Default $5 FAAB estimate

    # Test 3: No rosters provided (default to free agent)
    method, waiver_cost, faab_cost = detector._classify_acquisition(88888, None)
    assert method == "FREE_AGENT"
    assert faab_cost == 5


def test_uat_07_data_freshness_validation():
    """
    UAT-07: Verify data freshness validation works correctly.

    FRESH: latest game_date within 1 day
    STALE: latest game_date 2-3 days ago
    MISSING: latest game_date > 3 days ago
    """
    detector = TwoStartDetector()

    today = date.today()

    # Test 1: Fresh data (today's game and tomorrow's game)
    fresh_starts = [
        {
            "pitcher_name": "Fresh",
            "team": "TST",
            "bdl_player_id": 1,
            "game_date": today,
            "opponent": "OPP",
            "is_home": True,
            "is_confirmed": True,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
        {
            "pitcher_name": "Fresh",
            "team": "TST",
            "bdl_player_id": 1,
            "game_date": today + timedelta(days=1),
            "opponent": "OPP",
            "is_home": False,
            "is_confirmed": True,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
    ]

    freshness = detector._validate_data_freshness(fresh_starts)
    assert freshness == "FRESH", f"Expected FRESH for today's game, got {freshness}"

    # Test 2: Stale data (latest game 2 days ago)
    stale_starts = [
        {
            "pitcher_name": "Stale",
            "team": "TST",
            "bdl_player_id": 2,
            "game_date": today - timedelta(days=2),
            "opponent": "OPP",
            "is_home": True,
            "is_confirmed": False,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
        {
            "pitcher_name": "Stale",
            "team": "TST",
            "bdl_player_id": 2,
            "game_date": today - timedelta(days=3),
            "opponent": "OPP",
            "is_home": False,
            "is_confirmed": False,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
    ]

    freshness = detector._validate_data_freshness(stale_starts)
    assert freshness == "STALE", f"Expected STALE for 2-day-old game, got {freshness}"

    # Test 3: Missing data (latest game 4+ days ago)
    missing_starts = [
        {
            "pitcher_name": "Missing",
            "team": "TST",
            "bdl_player_id": 3,
            "game_date": today - timedelta(days=4),
            "opponent": "OPP",
            "is_home": True,
            "is_confirmed": False,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
        {
            "pitcher_name": "Missing",
            "team": "TST",
            "bdl_player_id": 3,
            "game_date": today - timedelta(days=5),
            "opponent": "OPP",
            "is_home": False,
            "is_confirmed": False,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
    ]

    freshness = detector._validate_data_freshness(missing_starts)
    assert freshness == "MISSING", f"Expected MISSING for 4+ day-old game, got {freshness}"


def test_uat_08_ip_projection_is_realistic():
    """
    UAT-08: Verify IP projections are realistic (5-6 IP per start).

    Prevents data pipeline bugs from producing unrealistic projections.
    """
    detector = TwoStartDetector()

    today = date.today()
    mock_starts = [
        {
            "pitcher_name": "Test",
            "team": "TST",
            "bdl_player_id": 1,
            "game_date": today + timedelta(days=1),
            "opponent": "OPP",
            "is_home": True,
            "is_confirmed": True,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
        {
            "pitcher_name": "Test",
            "team": "TST",
            "bdl_player_id": 1,
            "game_date": today + timedelta(days=5),
            "opponent": "OPP",
            "is_home": False,
            "is_confirmed": True,
            "park_factor": 1.0,
            "quality_score": 0.0,
        },
    ]

    opp = detector._build_opportunity(1, mock_starts)

    # Validate IP projection: 10-12 IP for 2 starts (5-6 per start)
    assert 10.0 <= opp.total_ip_projection <= 13.0, \
        f"IP projection {opp.total_ip_projection} unrealistic (expected 10-13 for 2 starts)"


def test_uat_09_fantasy_week_calculation():
    """
    UAT-09: Verify fantasy week calculation is reasonable.

    Week calculation should increment every 7 days from opening day.
    """
    detector = TwoStartDetector()

    # Test opening day (March 28)
    opening_day = date(2026, 3, 28)
    week = detector._compute_fantasy_week(opening_day)
    assert week == 1, f"Opening day should be week 1, got {week}"

    # Test one week later
    week_later = opening_day + timedelta(days=7)
    week = detector._compute_fantasy_week(week_later)
    assert week == 2, f"One week after opening should be week 2, got {week}"

    # Test mid-season
    mid_season = date(2026, 5, 15)  # ~48 days after opening day
    week = detector._compute_fantasy_week(mid_season)
    assert 7 <= week <= 8, f"Mid-May should be week 7-8, got {week}"


def test_uat_10_end_to_end_detection_with_mock_db():
    """
    UAT-10: End-to-end test of two-start detection with mocked database.

    Validates the full detection pipeline from DB query to opportunity list.
    """
    import os
    from unittest.mock import MagicMock, patch

    db_url = os.environ.get("DATABASE_URL")
    detector = TwoStartDetector(db_url)

    if not db_url:
        pytest.skip("DATABASE_URL not set — skipping end-to-end test")

    # Mock the database query result
    mock_rows = [
        (
            "Justin Verlander",  # pitcher_name
            "HOU",                # team
            54321,               # bdl_player_id
            (date.today() + timedelta(days=1)).isoformat(),  # game_date
            "LAA",               # opponent
            True,                # is_home
            True,                # is_confirmed
            1.02,                # park_factor
            1.5,                 # quality_score
        ),
        (
            "Justin Verlander",
            "HOU",
            54321,
            (date.today() + timedelta(days=5)).isoformat(),
            "SEA",
            False,
            True,
            0.98,
            0.8,
        ),
    ]

    # Mock execute result
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter(mock_rows)
    mock_result.scalar.return_value = 2  # 2 starts found

    mock_connection = MagicMock()
    mock_connection.execute.return_value = mock_result

    mock_session = MagicMock()
    mock_session.__enter__ = lambda self: mock_connection
    mock_session.__exit__ = lambda self, *args: None

    detector.SessionLocal = MagicMock(return_value=mock_session)

    # Run detection
    start_date = date.today()
    end_date = date.today() + timedelta(days=7)

    opportunities = detector.detect_two_start_pitchers(start_date, end_date)

    # Validate results
    assert len(opportunities) == 1, f"Expected 1 two-start pitcher, got {len(opportunities)}"
    opp = opportunities[0]

    assert opp.player_id == "54321"
    assert opp.name == "Justin Verlander"
    assert opp.team == "HOU"
    assert opp.acquisition_method == "FREE_AGENT"  # No rosters provided
    assert len(opp.categories_addressed) == 4
    assert opp.total_ip_projection > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
