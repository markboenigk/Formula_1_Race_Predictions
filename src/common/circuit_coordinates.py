"""
Shared circuit coordinates and event name normalization for F1 pipeline.

Provides a single source of truth for normalize_event_name() (used in S3 paths
and join keys) and lat/lng coordinates for all 2022–2025 calendar circuits.
No external dependencies beyond Python stdlib (re).
"""
import re


def normalize_event_name(event_name: str) -> str:
    """
    Keep consistent with Bronze `event=` slug creation.

    Example: "Monaco Grand Prix" -> "monaco"
    """
    normalized = str(event_name or "").replace("Grand Prix", "").strip()
    normalized = normalized.replace("São Paulo", "sao_paulo")
    normalized = normalized.replace("Mexico City", "mexico_city")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


# 25 circuits: 24 active 2025 + French GP (Paul Ricard). Keys from normalize_event_name().
CIRCUIT_COORDINATES: dict[str, tuple[float, float]] = {
    normalize_event_name("Bahrain Grand Prix"): (26.0325, 50.5106),
    normalize_event_name("Saudi Arabian Grand Prix"): (21.6319, 39.1044),
    normalize_event_name("Australian Grand Prix"): (-37.8497, 144.9680),
    normalize_event_name("Japanese Grand Prix"): (34.8431, 136.5407),
    normalize_event_name("Chinese Grand Prix"): (31.3389, 121.2197),
    normalize_event_name("Miami Grand Prix"): (25.9581, -80.2389),
    normalize_event_name("Emilia Romagna Grand Prix"): (44.3439, 11.7167),
    normalize_event_name("Monaco Grand Prix"): (43.7347, 7.4206),
    normalize_event_name("Canadian Grand Prix"): (45.5000, -73.5228),
    normalize_event_name("Spanish Grand Prix"): (41.5700, 2.2611),
    normalize_event_name("Austrian Grand Prix"): (47.2197, 14.7647),
    normalize_event_name("British Grand Prix"): (52.0786, -1.0169),
    normalize_event_name("Hungarian Grand Prix"): (47.5789, 19.2486),
    normalize_event_name("Belgian Grand Prix"): (50.4372, 5.9714),
    normalize_event_name("Dutch Grand Prix"): (52.3888, 4.5409),
    normalize_event_name("Italian Grand Prix"): (45.6156, 9.2811),
    normalize_event_name("Azerbaijan Grand Prix"): (40.3725, 49.8533),
    normalize_event_name("Singapore Grand Prix"): (1.2914, 103.8644),
    normalize_event_name("United States Grand Prix"): (30.1328, -97.6411),
    normalize_event_name("Mexico City Grand Prix"): (19.4042, -99.0907),
    normalize_event_name("Sao Paulo Grand Prix"): (-23.7036, -46.6997),
    normalize_event_name("Las Vegas Grand Prix"): (36.1162, -115.1745),
    normalize_event_name("Qatar Grand Prix"): (25.4900, 51.4542),
    normalize_event_name("Abu Dhabi Grand Prix"): (24.4672, 54.6031),
    normalize_event_name("French Grand Prix"): (43.2506, 5.7917),
}


def get_circuit_coords(event_name: str) -> tuple[float, float]:
    """Return (lat, lng) for a known circuit. Raises KeyError for unknown circuits."""
    slug = normalize_event_name(event_name)
    if slug not in CIRCUIT_COORDINATES:
        raise KeyError(f"No coordinates for circuit: '{event_name}' (slug: '{slug}')")
    return CIRCUIT_COORDINATES[slug]
