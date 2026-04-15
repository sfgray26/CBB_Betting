from datetime import date
from unittest.mock import MagicMock, patch


def test_load_probable_pitchers_from_snapshot_normalizes_keys():
    from backend.services.probable_pitcher_fallback import load_probable_pitchers_from_snapshot

    db = MagicMock()
    rows = [("TBR", "Shane Baz"), ("NYY", "Gerrit Cole")]
    db.query.return_value.filter.return_value.all.return_value = rows

    result = load_probable_pitchers_from_snapshot(db, date(2026, 4, 15))

    assert result["TB"] == "shane baz"
    assert result["NYY"] == "gerrit cole"


def test_lineup_optimizer_prefers_snapshot_before_live_api():
    from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer

    fake_db = MagicMock()
    optimizer = DailyLineupOptimizer()

    with patch("backend.fantasy_baseball.daily_lineup_optimizer.SessionLocal", return_value=fake_db):
        with patch(
            "backend.fantasy_baseball.daily_lineup_optimizer.load_probable_pitchers_from_snapshot",
            return_value={"NYY": "gerrit cole"},
        ) as snapshot_mock:
            with patch("backend.fantasy_baseball.daily_lineup_optimizer.requests.get") as requests_mock:
                result = optimizer._fetch_probable_pitchers_for_date("2026-04-15")

    assert result == {"NYY": "gerrit cole"}
    assert snapshot_mock.called is True
    requests_mock.assert_not_called()


def test_lineup_optimizer_uses_inference_if_snapshot_and_api_empty():
    from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer

    fake_db = MagicMock()
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"dates": []}

    optimizer = DailyLineupOptimizer()

    with patch("backend.fantasy_baseball.daily_lineup_optimizer.SessionLocal", return_value=fake_db):
        with patch(
            "backend.fantasy_baseball.daily_lineup_optimizer.load_probable_pitchers_from_snapshot",
            return_value={},
        ):
            with patch("backend.fantasy_baseball.daily_lineup_optimizer.requests.get", return_value=fake_response):
                with patch(
                    "backend.fantasy_baseball.daily_lineup_optimizer.infer_probable_pitcher_map",
                    return_value={"NYY": MagicMock(pitcher_name="Gerrit Cole")},
                ):
                    result = optimizer._fetch_probable_pitchers_for_date("2026-04-15")

    assert result == {"NYY": "gerrit cole"}