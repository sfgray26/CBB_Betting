"""
Tests for id_resolution_service.get_quarantined_identity_ids().
Uses unittest.mock to avoid a real DB session.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.fantasy_baseball.id_resolution_service import get_quarantined_identity_ids


class TestGetQuarantinedIdentityIds:

    def _make_session(self, rows):
        """Build a mock Session whose query chain returns `rows`."""
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.all.return_value = rows
        session.query.return_value = q
        return session

    def test_returns_empty_set_when_none_pending(self):
        session = self._make_session([])
        result = get_quarantined_identity_ids(session)
        assert result == set()

    def test_returns_proposed_ids(self):
        Row = MagicMock()
        Row.proposed_player_id = 42
        session = self._make_session([Row])
        result = get_quarantined_identity_ids(session)
        assert 42 in result

    def test_returns_multiple_ids(self):
        rows = [MagicMock(proposed_player_id=i) for i in [1, 2, 3]]
        session = self._make_session(rows)
        result = get_quarantined_identity_ids(session)
        assert result == {1, 2, 3}

    def test_returns_set_not_list(self):
        rows = [MagicMock(proposed_player_id=7)]
        session = self._make_session(rows)
        result = get_quarantined_identity_ids(session)
        assert isinstance(result, set)

    def test_deduplicates_ids(self):
        rows = [MagicMock(proposed_player_id=5), MagicMock(proposed_player_id=5)]
        session = self._make_session(rows)
        result = get_quarantined_identity_ids(session)
        assert result == {5}
