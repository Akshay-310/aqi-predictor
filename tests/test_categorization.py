"""Tests for AQI category boundary logic in app.py."""
import app as dashboard_app


def test_category_boundaries():
    cases = [
        (0, "Good"), (50, "Good"),
        (51, "Moderate"), (100, "Moderate"),
        (101, "Unhealthy (Sensitive)"), (150, "Unhealthy (Sensitive)"),
        (151, "Unhealthy"), (200, "Unhealthy"),
        (201, "Very Unhealthy"), (300, "Very Unhealthy"),
        (301, "Hazardous"), (500, "Hazardous"),
    ]
    for aqi, expected in cases:
        name, _ = dashboard_app.categorize(aqi)
        assert name == expected, f"AQI={aqi} expected '{expected}', got '{name}'"


def test_category_above_scale_falls_back_to_hazardous():
    name, _ = dashboard_app.categorize(750)
    assert name == "Hazardous"


def test_severity_index_is_monotonically_increasing():
    names = [c[2] for c in dashboard_app.CATEGORY_SCALE]
    indices = [dashboard_app.severity_index(n) for n in names]
    assert indices == sorted(indices)