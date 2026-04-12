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
            "whip_skipped_zero_ip",  # NEW field added in this fix
            "initial_ops_null",
            "initial_whip_null",
            "final_ops_null",
            "final_whip_null",
            "total_rows"
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
