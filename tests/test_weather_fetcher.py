from datetime import datetime
from unittest.mock import Mock

import requests

from backend.fantasy_baseball.weather_fetcher import WeatherFetcher


def test_estimate_weather_cold_conditions_reduce_hitter_score():
    fetcher = WeatherFetcher(api_key=None)

    weather = fetcher._estimate_weather(
        venue="Wrigley Field",
        game_time=datetime(2026, 4, 15, 13, 20),
        stadium_profile={"park_factor": 1.0, "elevation": 594},
    )

    assert weather.fallback_mode is True
    assert weather.temperature == 58
    assert weather.hitter_friendly_score < 5.0
    assert weather.hr_factor < 1.0


def test_estimate_weather_hot_conditions_boost_hitter_score():
    fetcher = WeatherFetcher(api_key=None)

    weather = fetcher._estimate_weather(
        venue="Truist Park",
        game_time=datetime(2026, 7, 15, 19, 20),
        stadium_profile={"park_factor": 1.02, "elevation": 1050},
    )

    assert weather.fallback_mode is True
    assert weather.temperature == 89
    assert weather.hitter_friendly_score > 5.0
    assert weather.hr_factor > 1.0


def test_openweather_401_falls_back_to_estimate(monkeypatch):
    fetcher = WeatherFetcher(api_key="bad-key")

    geo_response = Mock()
    geo_response.status_code = 200
    geo_response.raise_for_status.return_value = None
    geo_response.json.return_value = [{"lat": 41.9484, "lon": -87.6553}]

    weather_response = Mock()
    weather_response.status_code = 401
    weather_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")

    monkeypatch.setattr(fetcher._session, "get", Mock(side_effect=[geo_response, weather_response]))

    weather = fetcher._fetch_openweather(
        venue="Wrigley Field",
        game_time=datetime(2026, 4, 15, 13, 20),
        city="Chicago",
        state="IL",
        stadium_profile={"park_factor": 1.0, "elevation": 594},
    )

    assert weather.fallback_mode is True
    assert fetcher._api_key_failed is True
    assert weather.temperature == 58