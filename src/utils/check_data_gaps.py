# src/utils/check_data_gaps.py
"""
Reads aqi_raw_hourly from Hopsworks, sorts by time, and reports any
gaps where consecutive hourly readings are more than 1 hour apart —
these represent missing hours in the dataset.
"""
import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()


def connect_to_hopsworks():
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
    )
    return project.get_feature_store()


def main():
    fs = connect_to_hopsworks()
    raw_fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
    df = raw_fg.read()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")

    # Check for duplicate timestamps
    dupes = df[df.duplicated(subset=["time"], keep=False)]
    print(f"Duplicate timestamps: {len(dupes)}")

    # Check for gaps — consecutive rows should be exactly 1 hour apart
    df["time_diff"] = df["time"].diff()
    expected = pd.Timedelta(hours=1)
    gaps = df[df["time_diff"] > expected]

    if gaps.empty:
        print("\nNo gaps found — every hour is present, fully continuous dataset.")
    else:
        print(f"\n{len(gaps)} gap(s) found:")
        for idx, row in gaps.iterrows():
            gap_start = df.loc[idx - 1, "time"]
            gap_end = row["time"]
            missing_hours = int(row["time_diff"].total_seconds() / 3600) - 1
            print(f"  Gap between {gap_start} and {gap_end}  →  {missing_hours} missing hour(s)")

    # Expected total rows if fully continuous, for a sanity check
    expected_rows = int((df["time"].max() - df["time"].min()).total_seconds() / 3600) + 1
    print(f"\nExpected rows if no gaps: {expected_rows}")
    print(f"Actual rows: {len(df)}")
    print(f"Missing: {expected_rows - len(df)}")


if __name__ == "__main__":
    main()