# src/preprocessing.py
"""
Data quality checks and cleaning applied to raw hourly data before it's
inserted into the feature store. Validates physical bounds, removes
duplicate timestamps, flags outliers, and handles short gaps.
"""
import pandas as pd
import numpy as np

# Physically valid ranges for each variable — values outside these are
# either sensor errors or API bugs, not real readings.
VALID_RANGES = {
    "us_aqi": (0, 500),
    "pm2_5": (0, 1000),
    "pm10": (0, 1000),
    "carbon_monoxide": (0, 50000),
    "nitrogen_dioxide": (0, 1000),
    "sulphur_dioxide": (0, 1000),
    "ozone": (0, 1000),
    "temperature_2m": (-10, 55),        # realistic range for Karachi
    "relative_humidity_2m": (0, 100),
    "wind_speed_10m": (0, 150),
    "wind_direction_10m": (0, 360),
    "surface_pressure": (850, 1100),
    "precipitation": (0, 500),
}


def remove_duplicate_timestamps(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate timestamp rows")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Flags out-of-range values as NaN rather than silently keeping bad data."""
    issues_found = 0
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        invalid_mask = (df[col] < low) | (df[col] > high)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            print(f"  {col}: {n_invalid} out-of-range values found (valid range: {low}-{high}), setting to NaN")
            df.loc[invalid_mask, col] = np.nan
            issues_found += n_invalid
    if issues_found == 0:
        print("  No out-of-range values found — data passed bounds validation")
    return df


def flag_outliers(df: pd.DataFrame, column: str = "us_aqi", z_thresh: float = 4.0) -> pd.DataFrame:
    """
    Flags statistical outliers using z-score, WITHOUT removing them —
    real pollution spikes are meaningful signal, not noise. This just
    logs them so you're aware, for your report.
    """
    mean, std = df[column].mean(), df[column].std()
    if std == 0 or pd.isna(std):
        return df
    z_scores = (df[column] - mean).abs() / std
    outliers = df[z_scores > z_thresh]
    if len(outliers) > 0:
        print(f"  {len(outliers)} statistical outliers flagged in '{column}' (|z| > {z_thresh}) — kept, not removed")
        print(f"  Outlier timestamps: {outliers['time'].tolist() if 'time' in df.columns else outliers.index.tolist()}")
    return df


def fill_short_gaps(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """
    Forward-fills missing values only for short gaps (e.g., a brief API
    hiccup in live collection). Longer gaps are left as NaN rather than
    filled, since forward-filling a long gap would fabricate fake flat
    data — same staleness concern we discussed for the live pipeline.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill(limit=max_gap_hours)
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full cleaning sequence, in order, with progress printed."""
    print("Running data quality checks...")
    df = remove_duplicate_timestamps(df)
    df = validate_ranges(df)
    df = flag_outliers(df, column="us_aqi")
    df = fill_short_gaps(df)
    print("Data quality checks complete.\n")
    return df