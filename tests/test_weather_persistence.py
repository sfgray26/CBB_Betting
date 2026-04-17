"""
Tests for weather and park factor persistence.
"""
import pytest
from datetime import date
from backend.models import WeatherForecast, ParkFactor
from backend.services.weather_ingestion import get_weather_orchestrator, DEFAULT_PARK_FACTORS


def test_weather_forecast_model_exists():
    """WeatherForecast model should exist with required fields."""
    assert hasattr(WeatherForecast, 'game_date')
    assert hasattr(WeatherForecast, 'park_name')
    assert hasattr(WeatherForecast, 'temperature_high')
    assert hasattr(WeatherForecast, 'conditions')


def test_park_factor_model_exists():
    """ParkFactor model should exist with required fields."""
    assert hasattr(ParkFactor, 'park_name')
    assert hasattr(ParkFactor, 'hr_factor')
    assert hasattr(ParkFactor, 'run_factor')
    assert hasattr(ParkFactor, 'data_source')


def test_default_park_factors_defined():
    """Default park factors should be defined for all MLB stadiums."""
    assert isinstance(DEFAULT_PARK_FACTORS, dict)
    assert len(DEFAULT_PARK_FACTORS) > 20  # Should have at least 26 parks

    # Check a few key parks exist
    assert "Coors Field" in DEFAULT_PARK_FACTORS
    assert "Yankee Stadium" in DEFAULT_PARK_FACTORS
    assert "Fenway Park" in DEFAULT_PARK_FACTORS

    # Coors should be very hitter-friendly
    assert DEFAULT_PARK_FACTORS["Coors Field"]["hr"] > 1.2


def test_weather_orchestrator_import():
    """WeatherOrchestrator should be importable."""
    from backend.services.weather_ingestion import WeatherOrchestrator
    assert WeatherOrchestrator is not None


def test_get_weather_orchestrator_singleton():
    """get_weather_orchestrator should return a singleton instance."""
    orch1 = get_weather_orchestrator()
    orch2 = get_weather_orchestrator()
    assert orch1 is orch2
