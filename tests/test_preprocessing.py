"""Tests for src/preprocessing.py's data quality/cleaning functions."""
import numpy as np
import pandas as pd
import preprocessing


def test_remove_duplicate_timestamps_keeps_first():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 01:00"]),
        "us_aqi": [50, 999, 60],  # duplicate row has a clearly-wrong value
    })
    result = preprocessing.remove_duplicate_timestamps(df)
    assert len(result) == 2
    assert result.loc[result["time"] == pd.Timestamp("2026-01-01 00:00"), "us_aqi"].iloc[0] == 50


def test_validate_ranges_flags_out_of_bounds_as_nan():
    df = pd.DataFrame({
        "us_aqi": [50, 999, -10],          # 999 and -10 are out of (0, 500)
        "temperature_2m": [25, 30, 100],   # 100 is out of (-10, 55)
    })
    result = preprocessing.validate_ranges(df.copy())
    assert result["us_aqi"].iloc[0] == 50
    assert pd.isna(result["us_aqi"].iloc[1])
    assert pd.isna(result["us_aqi"].iloc[2])
    assert pd.isna(result["temperature_2m"].iloc[2])
    assert result["temperature_2m"].iloc[0] == 25


def test_validate_ranges_leaves_valid_data_untouched():
    df = pd.DataFrame({"us_aqi": [10, 50, 100, 300]})
    result = preprocessing.validate_ranges(df.copy())
    assert result["us_aqi"].tolist() == [10, 50, 100, 300]


def test_flag_outliers_does_not_modify_data():
    """flag_outliers only logs — real pollution spikes are meaningful
    signal, not noise, and should never be silently removed."""
    values = [50] * 20 + [400]  # one clear spike
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=len(values), freq="h"),
        "us_aqi": values,
    })
    result = preprocessing.flag_outliers(df.copy(), column="us_aqi")
    assert result["us_aqi"].tolist() == values


def test_fill_short_gaps_fills_within_limit():
    df = pd.DataFrame({"us_aqi": [50.0, np.nan, np.nan, 60.0]})
    result = preprocessing.fill_short_gaps(df.copy(), max_gap_hours=3)
    assert result["us_aqi"].tolist() == [50.0, 50.0, 50.0, 60.0]


def test_fill_short_gaps_leaves_long_gaps_as_nan():
    df = pd.DataFrame({"us_aqi": [50.0] + [np.nan] * 5 + [60.0]})
    result = preprocessing.fill_short_gaps(df.copy(), max_gap_hours=3)
    # ffill(limit=3) only fills the first 3 NaNs after a valid value
    assert result["us_aqi"].iloc[1:4].tolist() == [50.0, 50.0, 50.0]
    assert result["us_aqi"].iloc[4:6].isna().all()