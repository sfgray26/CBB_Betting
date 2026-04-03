from datetime import datetime

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