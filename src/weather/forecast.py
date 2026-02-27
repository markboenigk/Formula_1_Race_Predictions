"""
Live race forecast utility for inference-time weather.

Fetches weather predictions from Open-Meteo's forecast API for an upcoming
race date. Used by the prediction Lambda in Phase 5 — not for historical
backfill. Caller is responsible for looking up circuit coordinates and
passing lat/lng (e.g. from get_circuit_coords).
"""
import requests

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _coalesce_precip(val) -> float:
    """Coalesce precipitation/probability to 0.0 if None."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def fetch_race_forecast(lat: float, lng: float, race_date: str) -> dict:
    """
    Fetch forecast weather for an upcoming race date.

    Uses Open-Meteo forecast API (up to 16 days ahead).
    Returns a dict with race-day weather fields matching the
    historical backfill schema where applicable, plus
    precipitation_probability_max (forecast-only field).

    Args:
        lat: Circuit latitude
        lng: Circuit longitude
        race_date: Race date in YYYY-MM-DD format (must be within 16-day forecast window)

    Returns:
        dict with keys:
            - precipitation_sum_mm (float)
            - precipitation_probability_max (float, 0-100 %)
            - temperature_max_c (float)
            - temperature_min_c (float)
            - wind_speed_max_kmh (float)

    Raises:
        requests.HTTPError: On non-200 response from Open-Meteo
        KeyError: If expected fields missing from API response
    """
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": race_date,
        "end_date": race_date,
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    daily = data["daily"]

    return {
        "precipitation_sum_mm": _coalesce_precip(daily.get("precipitation_sum", [None])[0] if daily.get("precipitation_sum") else None),
        "precipitation_probability_max": _coalesce_precip(daily.get("precipitation_probability_max", [None])[0] if daily.get("precipitation_probability_max") else None),
        "temperature_max_c": daily["temperature_2m_max"][0],
        "temperature_min_c": daily["temperature_2m_min"][0],
        "wind_speed_max_kmh": daily["wind_speed_10m_max"][0],
    }
