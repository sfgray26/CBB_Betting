"""
Test that backfill-ops-whip endpoint accurately counts actual updates.

These tests verify:
1. The endpoint excludes zero-inning rows from WHIP backfill
2. The endpoint reports the number of zero-IP rows skipped

Note: Full DB integration tests require a test database with zero-IP rows.
These tests verify the SQL logic structure and endpoint contract.
"""
import pytest


class TestBackfillOpsWhipSQLLogic:
    """Tests for the SQL logic in backfill-ops-whip endpoint."""

    def test_whip_update_excludes_zero_ip_from_sql(self):
        """
        Verify the WHIP UPDATE statement filters out zero-inning rows.

        The SQL should include: AND innings_pitched NOT IN ('0.0', '0', '0.00')
        This prevents counting NULL→NULL no-ops as "updated".
        """
        import inspect
        from backend import admin_backfill_ops_whip

        # Get the source code of the backfill function
        source = inspect.getsource(admin_backfill_ops_whip.backfill_ops_whip)

        # Verify the zero-IP exclusion filter exists
        assert "innings_pitched NOT IN" in source, \
            "WHIP UPDATE should exclude zero-inning rows with NOT IN clause"

        # Verify specific zero values are excluded
        assert "'0.0'" in source, "Should exclude '0.0' innings_pitched"
        # The filter might use different formats, check at least one exists
        has_zero_exclusion = (
            "NOT IN ('0.0'" in source or
            "NOT IN ('0'" in source or
            "!= '0.0'" in source
        )
        assert has_zero_exclusion, "Should have some form of zero-IP exclusion"

    def test_whip_skipped_zero_ip_field_exists(self):
        """Verify the endpoint includes whip_skipped_zero_ip in result dict."""
        import inspect
        from backend import admin_backfill_ops_whip

        # Get the source code of the backfill function
        source = inspect.getsource(admin_backfill_ops_whip.backfill_ops_whip)

        # Verify the new diagnostic field exists
        assert 'whip_skipped_zero_ip' in source, \
            "Result dict should include whip_skipped_zero_ip diagnostic field"

    def test_whip_update_includes_division_by_zero_protection(self):
        """
        Verify the WHIP UPDATE uses NULLIF to prevent division by zero.

        Even with the innings_pitched filter, NULLIF provides defense-in-depth.
        """
        import inspect
        from backend import admin_backfill_ops_whip

        source = inspect.getsource(admin_backfill_ops_whip.backfill_ops_whip)

        # Verify NULLIF is used for division-by-zero protection
        assert "NULLIF" in source, \
            "WHIP calculation should use NULLIF to prevent division by zero"


class TestBackfillOpsWhipContract:
    """Tests for the endpoint response contract."""

    def test_backfill_function_exists_and_callable(self):
        """Verify the backfill function can be imported and called."""
        from backend import admin_backfill_ops_whip

        assert hasattr(admin_backfill_ops_whip, 'backfill_ops_whip')
        assert callable(admin_backfill_ops_whip.backfill_ops_whip)

    def test_result_dict_has_all_required_fields(self):
        """Verify the result dict includes all expected fields."""
        import inspect
        from backend import admin_backfill_ops_whip

        source = inspect.getsource(admin_backfill_ops_whip.backfill_ops_whip)

        # Required fields in the result dict
        required_fields = [
            "status",
            "ops_updated",
            "whip_updated",
            "whip_skipped_zero_ip",
            "caught_stealing_updated",
            "initial_ops_null",
            "initial_whip_null",
            "initial_cs_null",
            "final_ops_null",
            "final_whip_null",
            "final_cs_null",
            "total_rows",
        ]

        for field in required_fields:
            assert f'"{field}"' in source or f"'{field}'" in source, \
                f"Result dict should include field: {field}"


class TestWhIPComputationLogic:
    """Tests for WHIP computation edge cases."""

    def test_whip_undefined_for_zero_innings(self):
        """
        WHIP = (BB + H) / IP is mathematically undefined when IP = 0.

        This test documents WHY we must exclude zero-inning rows:
        Division by zero would produce NULL or infinite values.
        """
        # Example: Pitcher enters game, gets 0 outs, leaves
        walks_allowed = 1
        hits_allowed = 1
        innings_pitched = 0.0

        # WHIP = (1 + 1) / 0.0 = undefined (division by zero)
        # In SQL: (2)::numeric / NULLIF(0.0, 0) = NULL
        # Without NULLIF: would cause division by zero error
        # With NULLIF(0.0, 0): returns NULL (2 / NULL = NULL)

        # The correct behavior is to leave WHIP as NULL for zero-inning appearances
        assert innings_pitched == 0.0
        # We cannot compute WHIP for zero innings - must remain NULL

    def test_whip_defined_for_nonzero_innings(self):
        """
        WHIP is valid when IP > 0.

        This confirms the formula works correctly for normal cases.
        """
        walks_allowed = 3
        hits_allowed = 6
        innings_pitched = 7.0

        expected_whip = (walks_allowed + hits_allowed) / innings_pitched
        assert abs(expected_whip - 1.286) < 0.001  # (3+6)/7 = 1.286


class TestBackfillCaughtStealing:
    """Tests for the /admin/backfill-caught-stealing endpoint."""

    def test_endpoint_function_exists(self):
        """Verify backfill_caught_stealing is importable and callable."""
        from backend import admin_backfill_ops_whip
        assert hasattr(admin_backfill_ops_whip, 'backfill_caught_stealing')
        assert callable(admin_backfill_ops_whip.backfill_caught_stealing)

    def test_result_dict_has_required_fields(self):
        """Verify the CS backfill result dict has the expected shape."""
        import inspect
        from backend import admin_backfill_ops_whip

        source = inspect.getsource(admin_backfill_ops_whip.backfill_caught_stealing)
        for field in ("status", "initial_null", "rows_updated", "final_null", "total_rows"):
            assert f'"{field}"' in source or f"'{field}'" in source, \
                f"Result dict should include field: {field}"

    def test_sql_sets_cs_to_zero(self):
        """
        The SQL must SET caught_stealing = 0 WHERE caught_stealing IS NULL.

        0 is correct because BDL does not return caught_stealing at all;
        the daily ingestion already defaults new rows to 0 -- this backfill
        aligns the historic rows with the same convention.
        """
        import inspect
        from backend import admin_backfill_ops_whip

        source = inspect.getsource(admin_backfill_ops_whip.backfill_caught_stealing)
        assert "caught_stealing = 0" in source, \
            "Backfill should SET caught_stealing = 0"
        assert "caught_stealing IS NULL" in source, \
            "Backfill should only update rows WHERE caught_stealing IS NULL"

    def test_ops_whip_endpoint_includes_cs_backfill(self):
        """
        POST /admin/backfill-ops-whip now also patches caught_stealing.

        This test verifies the combined endpoint runs the CS UPDATE so callers
        only need one endpoint call to fix all three NULL columns.
        """
        import inspect
        from backend import admin_backfill_ops_whip

        source = inspect.getsource(admin_backfill_ops_whip.backfill_ops_whip)
        assert "caught_stealing = 0" in source, \
            "backfill_ops_whip should also SET caught_stealing = 0"
        assert "caught_stealing_updated" in source, \
            "backfill_ops_whip should track rows updated for caught_stealing"

    def test_nsb_formula_works_with_zero_cs(self):
        """
        NSB = SB - CS is only meaningful when CS is not NULL.

        Defaulting CS to 0 means NSB = SB (no penalty), which slightly
        overestimates for the rare player who was caught, but it's far
        better than propagating NULL to the NSB computation.
        """
        # Without fix: cs = None -> SB - None = None (NSB broken)
        sb = 5
        cs_null = None
        nsb_broken = sb - cs_null if cs_null is not None else None
        assert nsb_broken is None

        # With fix: cs = 0 -> SB - 0 = SB (NSB correct for most players)
        cs_fixed = 0
        nsb_fixed = sb - cs_fixed
        assert nsb_fixed == 5


class TestBackfillCsFromStatcast:
    """Tests for /admin/backfill-cs-from-statcast — Statcast-sourced CS truth."""

    def test_endpoint_function_exists(self):
        from backend import admin_backfill_ops_whip
        assert hasattr(admin_backfill_ops_whip, 'backfill_cs_from_statcast')
        assert callable(admin_backfill_ops_whip.backfill_cs_from_statcast)

    def test_result_dict_has_required_fields(self):
        import inspect
        from backend import admin_backfill_ops_whip
        source = inspect.getsource(admin_backfill_ops_whip.backfill_cs_from_statcast)
        for field in (
            "status",
            "total_rows",
            "cs_positive_before",
            "cs_positive_after",
            "rows_updated",
            "statcast_cs_events",
            "unmatched_statcast_events",
        ):
            assert f'"{field}"' in source or f"'{field}'" in source, \
                f"Result dict should include field: {field}"

    def test_joins_statcast_via_player_id_mapping(self):
        """The join must go through player_id_mapping — mlbam_id (Statcast) ↔ bdl_id (MPS)."""
        import inspect
        from backend import admin_backfill_ops_whip
        source = inspect.getsource(admin_backfill_ops_whip.backfill_cs_from_statcast)
        assert "statcast_performances" in source
        assert "player_id_mapping" in source
        assert "mlbam_id" in source and "bdl_id" in source, \
            "Join must traverse both mlbam_id and bdl_id columns"
        assert "game_date" in source, "Must match on game_date"

    def test_only_updates_positive_cs(self):
        """
        Only overwrite when Statcast recorded a real CS event (cs > 0).
        Prior =0 default is retained for player-games with no Statcast match.
        """
        import inspect
        from backend import admin_backfill_ops_whip
        source = inspect.getsource(admin_backfill_ops_whip.backfill_cs_from_statcast)
        assert "sp.cs > 0" in source, "Must filter to rows where Statcast cs > 0"

    def test_update_is_idempotent(self):
        """Re-running should be a no-op when caught_stealing already matches sp.cs."""
        import inspect
        from backend import admin_backfill_ops_whip
        source = inspect.getsource(admin_backfill_ops_whip.backfill_cs_from_statcast)
        assert "caught_stealing <> sp.cs" in source or "caught_stealing != sp.cs" in source, \
            "Should skip rows where caught_stealing already equals sp.cs"

    def test_mlbam_id_cast_to_text(self):
        """statcast_performances.player_id is STRING; mapping.mlbam_id is INT — cast required."""
        import inspect
        from backend import admin_backfill_ops_whip
        source = inspect.getsource(admin_backfill_ops_whip.backfill_cs_from_statcast)
        assert "mlbam_id::text" in source, \
            "Must cast mlbam_id to text to match statcast_performances.player_id"
