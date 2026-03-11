"""
Tests for the integrity sweep pipeline v3.0:
  - _ddgs_and_check()        — async DDGS + integrity check
  - _integrity_sweep()       — concurrent sweep over BET-tier games

All external I/O (DDGS) is mocked. No live network calls needed.

Patch targets:
  - duckduckgo_search.DDGS — use source path (imported inside function)
  - backend.services.analysis.async_perform_sanity_check — use where imported
"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from backend.services.analysis import (
    _ddgs_and_check,
    _integrity_sweep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _game(n=0, edge=0.05):
    return {
        "game_key": f"Away{n}@Home{n}",
        "home_team": f"Home{n}",
        "away_team": f"Away{n}",
        "edge": edge,
    }


def _ddgs_ctx(results=None):
    """Build a mock DDGS context manager that returns given results from .text()."""
    if results is None:
        results = [{"body": "No injuries reported."}]
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = results
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_ddgs)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# _ddgs_and_check (async)
# ---------------------------------------------------------------------------

class TestDdgsAndCheckAsync:

    def test_is_coroutine_function(self):
        import inspect
        assert inspect.iscoroutinefunction(_ddgs_and_check)

    @pytest.mark.asyncio
    async def test_happy_path_returns_verdict(self):
        """DDGS returns results, integrity check returns verdict — function returns it."""
        ctx = _ddgs_ctx([{"body": "No injuries."}])
        
        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.analysis.async_perform_sanity_check", 
                   new_callable=AsyncMock, return_value="CONFIRMED") as mock_check:
            
            result = await _ddgs_and_check(_game())

        assert result == "CONFIRMED"
        mock_check.assert_called_once()
        # Check that game_key was passed for escalation tracking
        call_kwargs = mock_check.call_args[1]
        assert call_kwargs.get("game_key") == "Away0@Home0"

    @pytest.mark.asyncio
    async def test_ddgs_exception_returns_graceful_fallback(self):
        """DDGS raises a RuntimeError — no crash, returns fallback string."""
        with patch("duckduckgo_search.DDGS", side_effect=RuntimeError("rate limited")):
            result = await _ddgs_and_check(_game())

        assert "Sanity check unavailable" in result

    @pytest.mark.asyncio
    async def test_empty_ddgs_results_calls_check_with_empty_context(self):
        """Empty results list produces empty context — check still called."""
        ctx = _ddgs_ctx([])
        captured = {}

        async def capture(*args, **kwargs):
            captured["context"] = kwargs.get("search_results")
            return "CONFIRMED"

        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.analysis.async_perform_sanity_check", 
                   side_effect=capture):

            result = await _ddgs_and_check(_game())

        assert result == "CONFIRMED"
        assert captured["context"] == ""

    @pytest.mark.asyncio
    async def test_verdict_placeholder_includes_edge_percentage(self):
        """Edge value from game dict appears in the verdict string sent to check."""
        ctx = _ddgs_ctx()
        captured = {}

        async def capture(*args, **kwargs):
            captured["verdict"] = kwargs.get("verdict")
            return "CONFIRMED"

        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.analysis.async_perform_sanity_check",
                   side_effect=capture):

            await _ddgs_and_check(_game(edge=0.12))

        assert "12.0%" in captured["verdict"]

    @pytest.mark.asyncio
    async def test_runs_concurrently(self):
        """5 tasks each sleeping 50ms should complete in ~50ms (not 250ms serial)."""

        async def slow_check(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate async I/O
            return "CONFIRMED"

        with patch("duckduckgo_search.DDGS", return_value=_ddgs_ctx()), \
             patch("backend.services.analysis.async_perform_sanity_check",
                   side_effect=slow_check):
            
            games = [_game(i) for i in range(5)]
            start = time.monotonic()
            results = await asyncio.gather(*[_ddgs_and_check(g) for g in games])
            elapsed = time.monotonic() - start

        assert all(r == "CONFIRMED" for r in results)
        # Serial would be 5 * 0.05 = 0.25s. Concurrent should be ~0.05s + overhead.
        assert elapsed < 0.20, f"Expected concurrent execution (<0.20s), got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# _integrity_sweep
# ---------------------------------------------------------------------------

class TestIntegritySweep:

    def test_empty_input_returns_empty_dict(self):
        result = asyncio.run(_integrity_sweep([]))
        assert result == {}

    def test_returns_dict_keyed_by_game_key(self):
        games = [_game(i) for i in range(3)]
        
        async def mock_check(game):
            return "CONFIRMED"
        
        with patch("backend.services.analysis._ddgs_and_check", side_effect=mock_check):
            result = asyncio.run(_integrity_sweep(games))

        assert set(result.keys()) == {"Away0@Home0", "Away1@Home1", "Away2@Home2"}
        assert all(v == "CONFIRMED" for v in result.values())

    def test_one_failing_game_does_not_abort_others(self):
        """If one game's check raises, the rest still produce verdicts."""
        games = [_game(i) for i in range(3)]
        call_count = 0

        async def flaky(game):
            nonlocal call_count
            call_count += 1
            if game["game_key"] == "Away1@Home1":
                raise RuntimeError("network blip")
            return "CONFIRMED"

        with patch("backend.services.analysis._ddgs_and_check", side_effect=flaky):
            result = asyncio.run(_integrity_sweep(games))

        assert len(result) == 3
        assert result["Away0@Home0"] == "CONFIRMED"
        assert result["Away2@Home2"] == "CONFIRMED"
        # Exception result gets replaced with the fallback string
        assert "Sanity check unavailable" in str(result["Away1@Home1"])

    def test_volatile_verdict_preserved(self):
        """VOLATILE verdicts from the integrity check flow through unchanged."""
        games = [_game(0), _game(1)]
        verdicts = iter(["VOLATILE", "CONFIRMED"])

        async def mock_check(game):
            return next(verdicts)

        with patch("backend.services.analysis._ddgs_and_check", side_effect=mock_check):
            result = asyncio.run(_integrity_sweep(games))

        volatile_vals = [v for v in result.values() if "VOLATILE" in str(v).upper()]
        assert len(volatile_vals) == 1

    def test_more_than_semaphore_limit_does_not_crash(self):
        """12 games (> semaphore limit of 8) complete without error."""
        games = [_game(i) for i in range(12)]
        
        async def mock_check(game):
            return "CONFIRMED"
        
        with patch("backend.services.analysis._ddgs_and_check", side_effect=mock_check):
            result = asyncio.run(_integrity_sweep(games))

        assert len(result) == 12
        assert all(v == "CONFIRMED" for v in result.values())

    def test_all_none_edges_handled(self):
        """Games with missing edge value don't crash the function."""
        game = {"game_key": "X@Y", "home_team": "Y", "away_team": "X"}  # no 'edge'
        ctx = _ddgs_ctx()
        
        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.analysis.async_perform_sanity_check",
                   new_callable=AsyncMock, return_value="CONFIRMED"):
            result = asyncio.run(_ddgs_and_check(game))

        assert result == "CONFIRMED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
