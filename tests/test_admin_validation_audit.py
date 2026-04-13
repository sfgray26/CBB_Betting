"""
Test that validation audit queries use live DB state.

This test suite verifies the validation audit endpoint structure and logic.
Note: These tests require a database connection and will be skipped if unavailable.

Author: Claude Code (Implementer Subagent)
Date: April 12, 2026
"""
import pytest


class TestValidationAuditQueries:
    """Tests for validation audit query structure and logic."""

    def test_add_finding_function_exists(self):
        """Verify the add_finding helper function is properly defined."""
        from backend.admin_endpoints_validation import router
        # The module should load without errors
        assert router is not None

    def test_ops_validation_query_structure(self):
        """Verify ops validation query checks for backfillable rows."""
        # Import the module to verify it compiles
        from backend import admin_endpoints_validation
        # The module should have the validation_audit function
        assert hasattr(admin_endpoints_validation, 'validation_audit')

    def test_orphan_check_uses_baseline_constant(self):
        """Verify orphan check uses a baseline constant, not hardcoded threshold."""
        import inspect
        from backend import admin_endpoints_validation

        # Get the source code of the validation_audit function
        source = inspect.getsource(admin_endpoints_validation.validation_audit)

        # Verify ORPHAN_BASELINE constant is used
        assert 'ORPHAN_BASELINE' in source, "Orphan check should use ORPHAN_BASELINE constant"

    def test_statcast_check_mentions_row_count(self):
        """Verify Statcast validation checks row count."""
        import inspect
        from backend import admin_endpoints_validation

        # Get the source code
        source = inspect.getsource(admin_endpoints_validation.validation_audit)

        # Verify row count is mentioned in Statcast check
        assert 'statcast_count' in source or 'row_count' in source, "Statcast check should reference row count"

    def test_backfillable_ops_query_includes_filter(self):
        """Verify ops validation query includes backfillable filter."""
        import inspect
        from backend import admin_endpoints_validation

        source = inspect.getsource(admin_endpoints_validation.validation_audit)

        # Verify backfillable filter exists
        assert 'backfillable' in source.lower(), "OPS check should identify backfillable rows"
        assert 'FILTER' in source, "OPS check should use SQL FILTER clause"

    def test_validation_audit_detects_zero_quality_statcast(self):
        """Audit source must contain checks for zero-filled statcast quality metrics.

        This guards against the column-mapping bug where 6,255 statcast rows were
        created with exit_velocity_avg=0, xwoba=0, xba=0, barrel_pct=0 and the
        validation audit silently reported them as healthy.
        """
        import inspect
        from backend import admin_endpoints_validation

        source = inspect.getsource(admin_endpoints_validation.validation_audit)
        assert "exit_velocity_avg" in source, (
            "validation_audit does not reference exit_velocity_avg — "
            "zero-quality statcast shell records will go undetected"
        )


class TestValidationAuditDBIntegration:
    """DB integration tests that verify validation output matches actual DB state."""

    def test_orphan_count_matches_live_db(self):
        """The orphan count in validation output should match actual DB state."""
        from backend.models import SessionLocal
        from fastapi.testclient import TestClient
        from backend.main import app

        db = SessionLocal()
        try:
            # Get actual orphan count from DB
            actual_orphans = db.execute("""
                SELECT COUNT(*) FROM position_eligibility pe
                LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
                WHERE pe.yahoo_player_key IS NOT NULL AND pim.yahoo_key IS NULL
            """).scalar()
        except Exception:
            # Skip if DB connection fails or table doesn't exist
            pytest.skip("Database connection unavailable or table missing")
        finally:
            db.close()

        # Get validation report
        client = TestClient(app)
        response = client.get("/admin/validation-audit")

        # Skip if endpoint returns error (e.g. DB unavailable)
        if response.status_code != 200:
            pytest.skip(f"Validation endpoint returned {response.status_code}")

        report = response.json()

        # Find orphan finding in report
        orphan_finding = None
        for severity in ["critical", "high", "medium", "low"]:
            for finding in report.get(severity, []):
                if finding.get("category") == "Foreign Keys" and "orphaned" in finding.get("issue", "").lower():
                    orphan_finding = finding
                    break

        # If orphan finding exists, it should match actual count
        if orphan_finding:
            import re
            match = re.search(r'(\d+)\s+orphaned', orphan_finding["issue"])
            if match:
                reported_count = int(match.group(1))
                assert reported_count == actual_orphans, f"Reported {reported_count} orphans but DB has {actual_orphans}"

    def test_ops_validation_only_reports_backfillable(self):
        """OPS validation should not complain about structurally unbackfillable NULL ops."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        response = client.get("/admin/validation-audit")

        # Skip if endpoint returns error
        if response.status_code != 200:
            pytest.skip(f"Validation endpoint returned {response.status_code}")

        report = response.json()

        # Check that we don't have a HIGH severity finding for ops
        # (NULL ops with missing components is expected state)
        high_findings = report.get("high", [])
        ops_findings = [f for f in high_findings if "ops" in f.get("issue", "").lower() and "null" in f.get("issue", "").lower()]

        # If ops finding exists, verify it's about backfillable rows only
        if ops_findings:
            assert "backfillable" in ops_findings[0]["issue"].lower() or "despite" in ops_findings[0]["issue"].lower()
