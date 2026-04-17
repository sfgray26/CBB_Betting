"""
Tests for /admin/version endpoint.

Tests verify the deployment fingerprint endpoint returns correct structure.
"""
import pytest


def test_admin_version_endpoint_exists():
    """GET /admin/version should exist and be registered."""
    from backend.main import app

    # Verify the route is registered
    routes = [route.path for route in app.routes]
    assert "/admin/version" in routes, "/admin/version endpoint should be registered"


def test_deployment_version_model_exists():
    """DeploymentVersion model should exist with required fields."""
    from backend.models import DeploymentVersion

    # Verify the model exists with required fields
    assert hasattr(DeploymentVersion, 'git_commit_sha')
    assert hasattr(DeploymentVersion, 'git_commit_date')
    assert hasattr(DeploymentVersion, 'build_timestamp')
    assert hasattr(DeploymentVersion, 'app_version')
