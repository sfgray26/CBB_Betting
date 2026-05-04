"""
Tests for GET /admin/explanations/{decision_id} endpoint auth.

Hardening validation: ensure decision explanation endpoint is properly protected.
"""
import pytest
import tempfile
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.models import DecisionExplanation, DecisionResult, get_db
from backend.auth import verify_api_key


# Mock auth that bypasses API key verification for tests
async def mock_verify_api_key():
    return "test_user"


@pytest.fixture
def client_with_explanation():
    """Test client with explanation preloaded."""
    temp_db = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
    temp_db.close()
    db_path = Path(temp_db.name)

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    DecisionResult.__table__.create(bind=engine)
    DecisionExplanation.__table__.create(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()

    # Add a decision result
    decision = DecisionResult(
        id=1,
        as_of_date=date(2026, 4, 15),
        decision_type="lineup",
        bdl_player_id=12345,
        target_slot="OF",
        drop_player_id=None,
        lineup_score=92.5,
        value_gain=8.3,
        confidence=0.87,
        reasoning="Strong recent form",
        computed_at=None,
    )
    db.add(decision)

    # Add explanation for that decision
    explanation = DecisionExplanation(
        id=1,
        decision_id=1,
        bdl_player_id=12345,
        as_of_date=date(2026, 4, 15),
        decision_type="lineup",
        summary="Strong recommendation to start",
        factors_json=[{"name": "z_score", "value": "1.85", "label": "Composite Z"}],
        confidence_narrative="High confidence",
        risk_narrative="Low risk",
        track_record_narrative="Strong track record",
        computed_at=None,
    )
    db.add(explanation)
    db.commit()

    def override_get_db():
        try:
            yield SessionLocal()
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = mock_verify_api_key

    client = TestClient(app)
    yield client

    db.close()
    app.dependency_overrides = {}
    try:
        db_path.unlink()
    except Exception:
        pass


def test_get_explanation_with_auth(client_with_explanation):
    """Authenticated request returns explanation data."""
    response = client_with_explanation.get("/admin/explanations/1")
    assert response.status_code == 200
    data = response.json()
    assert data["decision_id"] == 1
    assert data["bdl_player_id"] == 12345
    assert data["summary"] == "Strong recommendation to start"


def test_get_explanation_unauthorized():
    """Request without API key returns 401."""
    # Clear overrides to test real auth behavior
    app.dependency_overrides = {}

    client = TestClient(app)
    response = client.get("/admin/explanations/1")
    assert response.status_code == 401
    assert "API key required" in response.json()["detail"]


def test_get_explanation_not_found(client_with_explanation):
    """Request for non-existent decision_id returns 404."""
    response = client_with_explanation.get("/admin/explanations/999")
    assert response.status_code == 404
    assert "No explanation for decision_id=999" in response.json()["detail"]
