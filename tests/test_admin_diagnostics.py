"""
Tests for GET /admin/diagnostics/field-coverage endpoint.

Verifies that the endpoint returns proper counts for scarcity_rank,
quality_score, and backfilled V31/V32 fields. Uses mocked DB to avoid
requiring live production data.
"""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.auth import verify_admin_api_key


# Mock auth that bypasses API key verification for tests
async def mock_verify_admin_api_key():
    return "test_user"


def test_field_coverage_returns_all_keys():
    """
    Field coverage endpoint returns all 4 table keys and proper structure.

    Verifies endpoint is accessible, returns HTTP 200, and has the
    expected JSON structure with status, as_of, and fields dict. Does NOT
    assert specific counts since test DB may not match production state.
    """
    # Override auth to skip API key verification
    app.dependency_overrides[verify_admin_api_key] = mock_verify_admin_api_key

    client = TestClient(app)
    response = client.get("/admin/diagnostics/field-coverage")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:200]}"
    data = response.json()

    # Top-level keys
    assert "status" in data
    assert "as_of" in data
    assert "fields" in data
    assert data["status"] == "ok"

    # All 4 table keys present
    fields = data["fields"]
    assert "position_eligibility" in fields
    assert "probable_pitchers" in fields
    assert "player_rolling_stats" in fields
    assert "player_scores" in fields

    # position_eligibility has all 3 expected keys
    pe = fields["position_eligibility"]
    assert "total" in pe
    assert "scarcity_rank_populated" in pe
    assert "league_rostered_pct_populated" in pe

    # probable_pitchers has both keys
    pp = fields["probable_pitchers"]
    assert "total" in pp
    assert "quality_score_populated" in pp

    # player_rolling_stats has all 3 keys
    prs = fields["player_rolling_stats"]
    assert "total" in prs
    assert "w_runs_populated" in prs
    assert "w_qs_populated" in prs

    # player_scores has all 3 keys
    ps = fields["player_scores"]
    assert "total" in ps
    assert "z_r_populated" in ps
    assert "z_k_p_populated" in ps

    # Values are integers (not None, not strings, not empty)
    assert isinstance(pe["total"], int)
    assert isinstance(pp["total"], int)
    assert isinstance(prs["total"], int)
    assert isinstance(ps["total"], int)


def _patched_session_execute(sql_text, rows_by_keyword):
    """Return a mock execute result whose fetchone/fetchall matches the SQL."""
    mock_db = MagicMock()

    def _side_effect(stmt, *args, **kwargs):
        sql = str(stmt)
        for keyword, rows in rows_by_keyword.items():
            if keyword in sql:
                mock_result = MagicMock()
                if isinstance(rows, list):
                    mock_result.fetchall.return_value = rows
                else:
                    mock_result.fetchone.return_value = rows
                # rowcount for UPDATE statements
                mock_result.rowcount = rows if isinstance(rows, int) else 0
                return mock_result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (0,)
        mock_result.fetchall.return_value = []
        mock_result.rowcount = 0
        return mock_result

    mock_db.execute.side_effect = _side_effect
    mock_db.commit.return_value = None
    mock_db.rollback.return_value = None
    mock_db.close.return_value = None
    return mock_db


def test_backfill_scarcity_rank_endpoint():
    """
    POST /admin/actions/backfill-scarcity-rank returns rows_updated and coverage list.
    """
    app.dependency_overrides[verify_admin_api_key] = mock_verify_admin_api_key

    with patch("backend.main.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db

        # UPDATE returns 150 updated rows
        update_result = MagicMock()
        update_result.rowcount = 150

        # Coverage query returns sample rows
        coverage_rows = [("C", 172, 172), ("SS", 270, 270), ("SP", 747, 747)]
        coverage_result = MagicMock()
        coverage_result.fetchall.return_value = coverage_rows

        mock_db.execute.side_effect = [update_result, coverage_result]
        mock_db.commit.return_value = None
        mock_db.close.return_value = None

        client = TestClient(app)
        response = client.post("/admin/actions/backfill-scarcity-rank")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert data["rows_updated"] == 150
    assert "coverage" in data
    assert isinstance(data["coverage"], list)
    assert data["coverage"][0]["position"] == "C"


def test_backfill_quality_scores_endpoint():
    """
    POST /admin/actions/backfill-quality-scores returns constraint status and null rows patched.
    """
    app.dependency_overrides[verify_admin_api_key] = mock_verify_admin_api_key

    with patch("backend.main.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db

        # Constraint already exists → ALTER TABLE raises exception
        alter_result = MagicMock()
        alter_result.rowcount = 0

        # Constraint exists check
        constraint_row = MagicMock()  # truthy → constraint present

        # UPDATE NULL rows: patched 61
        update_result = MagicMock()
        update_result.rowcount = 61

        # Summary counts
        summary_row = (61, 61, 0.0)

        mock_db.execute.side_effect = [
            Exception("constraint already exists"),  # ALTER TABLE fails → caught
            constraint_row,                          # constraint check
            update_result,                           # UPDATE quality_score
            MagicMock(fetchone=lambda: summary_row), # summary SELECT
        ]
        mock_db.commit.return_value = None
        mock_db.rollback.return_value = None
        mock_db.close.return_value = None

        client = TestClient(app)
        response = client.post("/admin/actions/backfill-quality-scores")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert "null_rows_patched" in data
    assert "constraint_present" in data


def test_patch_null_teams_endpoint():
    """
    POST /admin/actions/patch-null-teams returns rows_patched and remaining count.
    """
    app.dependency_overrides[verify_admin_api_key] = mock_verify_admin_api_key

    with patch("backend.main.SessionLocal") as mock_sl:
        mock_db = MagicMock()
        mock_sl.return_value = mock_db

        update_result = MagicMock()
        update_result.rowcount = 311

        remaining_result = MagicMock()
        remaining_result.fetchone.return_value = (0,)

        mock_db.execute.side_effect = [update_result, remaining_result]
        mock_db.commit.return_value = None
        mock_db.close.return_value = None

        client = TestClient(app)
        response = client.post("/admin/actions/patch-null-teams")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert data["rows_patched"] == 311
    assert data["remaining_null_team"] == 0
