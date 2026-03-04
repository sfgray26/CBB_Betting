"""
Tests for the integrity sweep pipeline:
  - _ddgs_and_check_sync()   — blocking DDGS + LLM sanity check
  - _ddgs_and_check()        — async thread-pool wrapper
  - _integrity_sweep()       — concurrent sweep over BET-tier games

All external I/O (DDGS, Ollama) is mocked. No live network calls needed.

Patch targets use the source-module paths because DDGS and perform_sanity_check
are imported inside the function body (lazy imports), so the patches must
target the originating modules, not backend.services.analysis.
  - duckduckgo_search.DDGS
  - backend.services.scout.perform_sanity_check
"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock

from backend.services.analysis import (
    _ddgs_and_check_sync,
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
# _ddgs_and_check_sync
# ---------------------------------------------------------------------------

class TestDdgsAndCheckSync:

    def test_happy_path_returns_scout_verdict(self):
        """DDGS returns results, scout returns verdict — function returns it."""
        ctx = _ddgs_ctx([{"body": "No injuries."}])
        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.scout.perform_sanity_check", return_value="CONFIRMED — all clear") as mock_sc:

            result = _ddgs_and_check_sync(_game())

        assert result == "CONFIRMED — all clear"
        mock_sc.assert_called_once()
        home, away, verdict, context = mock_sc.call_args[0]
        assert home == "Home0"
        assert away == "Away0"
        assert "No injuries" in context

    def test_ddgs_exception_returns_graceful_fallback(self):
        """DDGS raises a RuntimeError — no crash, returns fallback string."""
        with patch("duckduckgo_search.DDGS", side_effect=RuntimeError("rate limited")):
            result = _ddgs_and_check_sync(_game())

        assert "Sanity check unavailable" in result
        assert "RuntimeError" in result

    def test_scout_exception_returns_graceful_fallback(self):
        """DDGS succeeds but Ollama is unreachable — no crash, returns fallback."""
        ctx = _ddgs_ctx()
        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.scout.perform_sanity_check", side_effect=ConnectionError("Ollama down")):

            result = _ddgs_and_check_sync(_game())

        assert "Sanity check unavailable" in result

    def test_empty_ddgs_results_calls_scout_with_empty_context(self):
        """Empty results list produces empty context — scout still called."""
        ctx = _ddgs_ctx([])
        captured = {}

        def capture(home, away, verdict, context):
            captured["context"] = context
            return "CONFIRMED"

        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.scout.perform_sanity_check", side_effect=capture):

            result = _ddgs_and_check_sync(_game())

        assert result == "CONFIRMED"
        assert captured["context"] == ""

    def test_verdict_placeholder_includes_edge_percentage(self):
        """Edge value from game dict appears in the verdict string sent to scout."""
        ctx = _ddgs_ctx()
        captured = {}

        def capture(home, away, verdict, context):
            captured["verdict"] = verdict
            return "CONFIRMED"

        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.scout.perform_sanity_check", side_effect=capture):

            _ddgs_and_check_sync(_game(edge=0.12))

        assert "12.0%" in captured["verdict"]


# ---------------------------------------------------------------------------
# _ddgs_and_check  (async wrapper)
# ---------------------------------------------------------------------------

class TestDdgsAndCheckAsync:

    def test_is_coroutine_function(self):
        import inspect
        assert inspect.iscoroutinefunction(_ddgs_and_check)

    def test_returns_result_from_sync_function(self):
        """Async wrapper passes game through to sync function and returns its result."""
        game = _game(0)
        with patch("backend.services.analysis._ddgs_and_check_sync", return_value="CONFIRMED thread") as mock_sync:
            result = asyncio.run(_ddgs_and_check(game))

        assert result == "CONFIRMED thread"
        mock_sync.assert_called_once_with(game)

    def test_runs_concurrently_via_thread_pool(self):
        """5 tasks each sleeping 50ms should complete in ~50ms (not 250ms serial)."""

        def slow_sync(game):
            time.sleep(0.05)
            return "CONFIRMED"

        async def run():
            games = [_game(i) for i in range(5)]
            start = time.monotonic()
            with patch("backend.services.analysis._ddgs_and_check_sync", side_effect=slow_sync):
                results = await asyncio.gather(*[_ddgs_and_check(g) for g in games])
            elapsed = time.monotonic() - start
            return results, elapsed

        results, elapsed = asyncio.run(run())
        assert all(r == "CONFIRMED" for r in results)
        # Serial would be 5 * 0.05 = 0.25s. Thread pool should be ~0.05s.
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
        with patch("backend.services.analysis._ddgs_and_check_sync", return_value="CONFIRMED"):
            result = asyncio.run(_integrity_sweep(games))

        assert set(result.keys()) == {"Away0@Home0", "Away1@Home1", "Away2@Home2"}
        assert all(v == "CONFIRMED" for v in result.values())

    def test_one_failing_game_does_not_abort_others(self):
        """If one game's check raises, the rest still produce verdicts."""
        games = [_game(i) for i in range(3)]
        call_count = 0

        def flaky(game):
            nonlocal call_count
            call_count += 1
            if game["game_key"] == "Away1@Home1":
                raise RuntimeError("network blip")
            return "CONFIRMED"

        with patch("backend.services.analysis._ddgs_and_check_sync", side_effect=flaky):
            result = asyncio.run(_integrity_sweep(games))

        assert len(result) == 3
        assert result["Away0@Home0"] == "CONFIRMED"
        assert result["Away2@Home2"] == "CONFIRMED"
        # Exception result gets replaced with the fallback string
        assert "Sanity check unavailable" in str(result["Away1@Home1"])

    def test_volatile_verdict_preserved(self):
        """VOLATILE verdicts from the LLM flow through unchanged."""
        games = [_game(0), _game(1)]
        verdicts = iter(["VOLATILE — key player doubtful", "CONFIRMED"])

        def mock_sync(game):
            return next(verdicts)

        with patch("backend.services.analysis._ddgs_and_check_sync", side_effect=mock_sync):
            result = asyncio.run(_integrity_sweep(games))

        volatile_vals = [v for v in result.values() if "VOLATILE" in str(v).upper()]
        assert len(volatile_vals) == 1

    def test_more_than_semaphore_limit_does_not_crash(self):
        """12 games (> semaphore limit of 8) complete without error."""
        games = [_game(i) for i in range(12)]
        with patch("backend.services.analysis._ddgs_and_check_sync", return_value="CONFIRMED"):
            result = asyncio.run(_integrity_sweep(games))

        assert len(result) == 12
        assert all(v == "CONFIRMED" for v in result.values())

    def test_all_none_edges_handled(self):
        """Games with missing edge value don't crash the sync function."""
        game = {"game_key": "X@Y", "home_team": "Y", "away_team": "X"}  # no 'edge'
        ctx = _ddgs_ctx()
        with patch("duckduckgo_search.DDGS", return_value=ctx), \
             patch("backend.services.scout.perform_sanity_check", return_value="CONFIRMED"):
            result = _ddgs_and_check_sync(game)

        assert result == "CONFIRMED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
