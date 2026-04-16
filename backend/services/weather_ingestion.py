"""
Weather and Park Factor Ingestion Service.

Canonical persistence for Layer 2 environmental context.
"""

import logging
from datetime import date
from typing import Optional

from backend.models import WeatherForecast, ParkFactor, SessionLocal

logger = logging.getLogger(__name__)


# Default park factors from Fangraphs (2024 season data)
DEFAULT_PARK_FACTORS = {
    "Yankee Stadium": {"hr": 1.02, "run": 1.01, "hits": 1.00, "era": 0.99, "whip": 1.00},
    "Dodger Stadium": {"hr": 0.95, "run": 0.97, "hits": 0.98, "era": 1.01, "whip": 1.00},
    "Coors Field": {"hr": 1.25, "run": 1.15, "hits": 1.10, "era": 1.10, "whip": 1.05},
    "Fenway Park": {"hr": 1.08, "run": 1.05, "hits": 1.03, "era": 1.02, "whip": 1.01},
    "Wrigley Field": {"hr": 1.05, "run": 1.04, "hits": 1.03, "era": 1.01, "whip": 1.01},
    "Oracle Park": {"hr": 0.92, "run": 0.95, "hits": 0.96, "era": 0.98, "whip": 0.99},
    "Truist Park": {"hr": 0.99, "run": 1.00, "hits": 0.99, "era": 1.00, "whip": 1.00},
    "Petco Park": {"hr": 0.94, "run": 0.96, "hits": 0.97, "era": 1.00, "whip": 0.99},
    "Citizens Bank Park": {"hr": 1.09, "run": 1.06, "hits": 1.04, "era": 1.02, "whip": 1.01},
    "Great American Ball Park": {"hr": 1.15, "run": 1.08, "hits": 1.05, "era": 1.03, "whip": 1.02},
    "American Family Field": {"hr": 1.05, "run": 1.03, "hits": 1.02, "era": 0.99, "whip": 1.00},
    "PNC Park": {"hr": 0.97, "run": 0.98, "hits": 0.98, "era": 0.99, "whip": 1.00},
    "LoanDepot Park": {"hr": 0.96, "run": 0.97, "hits": 0.97, "era": 1.01, "whip": 1.00},
    "Citi Field": {"hr": 0.95, "run": 0.97, "hits": 0.97, "era": 1.00, "whip": 1.00},
    "Nationals Park": {"hr": 1.00, "run": 1.00, "hits": 1.00, "era": 1.00, "whip": 1.00},
    "Tropicana Field": {"hr": 0.94, "run": 0.96, "hits": 0.97, "era": 1.00, "whip": 0.99},
    "Busch Stadium": {"hr": 1.00, "run": 1.00, "hits": 1.00, "era": 1.00, "whip": 1.00},
    "Comerica Park": {"hr": 1.02, "run": 1.02, "hits": 1.01, "era": 1.00, "whip": 1.00},
    "Kauffman Stadium": {"hr": 1.00, "run": 1.00, "hits": 1.00, "era": 1.00, "whip": 1.00},
    "Target Field": {"hr": 0.98, "run": 0.99, "hits": 0.99, "era": 1.01, "whip": 1.00},
    "Globe Life Field": {"hr": 0.98, "run": 0.99, "hits": 0.99, "era": 1.00, "whip": 1.00},
    "Angel Stadium": {"hr": 0.98, "run": 0.99, "hits": 0.99, "era": 1.00, "whip": 1.00},
    "Oakland Coliseum": {"hr": 0.97, "run": 0.98, "hits": 0.98, "era": 1.01, "whip": 1.00},
    "Rogers Centre": {"hr": 1.03, "run": 1.02, "hits": 1.01, "era": 1.00, "whip": 1.00},
    "T-Mobile Park": {"hr": 0.94, "run": 0.96, "hits": 0.97, "era": 1.01, "whip": 1.00},
    "Progressive Field": {"hr": 1.02, "run": 1.02, "hits": 1.01, "era": 1.00, "whip": 1.00},
    "Guaranteed Rate Field": {"hr": 1.07, "run": 1.05, "hits": 1.04, "era": 1.01, "whip": 1.01},
}


class WeatherOrchestrator:
    """
    Service for persisting canonical weather and park factor context.

    This satisfies Criterion 6 of Layer 2 certification:
    "Weather and park context are persisted canonically rather than
    trapped in request-time logic."
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def seed_default_park_factors(self) -> dict:
        """
        Seed park_factors table with default Fangraphs data.

        Returns:
            {"seeded": N, "skipped": M}
        """
        result = {"seeded": 0, "skipped": 0}

        db = SessionLocal()
        try:
            for park_name, factors in DEFAULT_PARK_FACTORS.items():
                existing = db.query(ParkFactor).filter_by(park_name=park_name).first()

                if existing:
                    result["skipped"] += 1
                    continue

                factor = ParkFactor(
                    park_name=park_name,
                    hr_factor=factors["hr"],
                    run_factor=factors["run"],
                    hits_factor=factors["hits"],
                    era_factor=factors["era"],
                    whip_factor=factors["whip"],
                    data_source="fangraphs",
                    season=2025
                )
                db.add(factor)
                result["seeded"] += 1

            db.commit()
            self._logger.info(f"Park factors seeded: {result}")
        finally:
            db.close()

        return result

    def get_park_factor(self, park_name: str) -> Optional[ParkFactor]:
        """
        Get park factor for a specific stadium.

        Args:
            park_name: Name of the stadium

        Returns:
            ParkFactor object or None if not found
        """
        db = SessionLocal()
        try:
            return db.query(ParkFactor).filter_by(park_name=park_name).first()
        finally:
            db.close()

    def upsert_weather_forecast(
        self,
        game_date: date,
        park_name: str,
        temperature_high: Optional[float] = None,
        temperature_low: Optional[float] = None,
        humidity: Optional[int] = None,
        wind_speed: Optional[float] = None,
        wind_direction: Optional[str] = None,
        precipitation_probability: Optional[int] = None,
        conditions: Optional[str] = None
    ) -> WeatherForecast:
        """
        Persist a weather forecast for a game.

        Args:
            game_date: Date of the game
            park_name: Stadium name
            temperature_high: High temperature in Celsius
            temperature_low: Low temperature in Celsius
            humidity: Humidity percentage
            wind_speed: Wind speed in km/h
            wind_direction: Wind direction (N, NE, E, etc.)
            precipitation_probability: Precipitation chance percentage
            conditions: Weather conditions description

        Returns:
            The created/updated WeatherForecast object
        """
        db = SessionLocal()
        try:
            # Check for existing forecast
            forecast = db.query(WeatherForecast).filter(
                WeatherForecast.game_date == game_date,
                WeatherForecast.park_name == park_name,
                WeatherForecast.forecast_date == date.today()
            ).first()

            if forecast:
                # Update existing
                if temperature_high is not None:
                    forecast.temperature_high = temperature_high
                if temperature_low is not None:
                    forecast.temperature_low = temperature_low
                if humidity is not None:
                    forecast.humidity = humidity
                if wind_speed is not None:
                    forecast.wind_speed = wind_speed
                if wind_direction is not None:
                    forecast.wind_direction = wind_direction
                if precipitation_probability is not None:
                    forecast.precipitation_probability = precipitation_probability
                if conditions is not None:
                    forecast.conditions = conditions
            else:
                # Create new
                forecast = WeatherForecast(
                    game_date=game_date,
                    park_name=park_name,
                    forecast_date=date.today(),
                    temperature_high=temperature_high,
                    temperature_low=temperature_low,
                    humidity=humidity,
                    wind_speed=wind_speed,
                    wind_direction=wind_direction,
                    precipitation_probability=precipitation_probability,
                    conditions=conditions
                )
                db.add(forecast)

            db.commit()
            db.refresh(forecast)
            return forecast
        finally:
            db.close()

    def get_weather_forecast(self, game_date: date, park_name: str) -> Optional[WeatherForecast]:
        """
        Get weather forecast for a game.

        Args:
            game_date: Date of the game
            park_name: Stadium name

        Returns:
            WeatherForecast object or None if not found
        """
        db = SessionLocal()
        try:
            return db.query(WeatherForecast).filter(
                WeatherForecast.game_date == game_date,
                WeatherForecast.park_name == park_name
            ).order_by(WeatherForecast.forecast_date.desc()).first()
        finally:
            db.close()


# Singleton instance
_weather_orchestrator = None

def get_weather_orchestrator() -> WeatherOrchestrator:
    """Get the singleton WeatherOrchestrator instance."""
    global _weather_orchestrator
    if _weather_orchestrator is None:
        _weather_orchestrator = WeatherOrchestrator()
    return _weather_orchestrator
