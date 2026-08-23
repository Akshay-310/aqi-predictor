import requests

test_dates = ["2022-03-01", "2022-06-01", "2022-09-01", "2022-12-01", "2023-03-01"]

for date in test_dates:
    resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": 24.8607, "longitude": 67.0011,
            "hourly": "us_aqi",
            "start_date": date,
            "end_date": date,
            "timezone": "Asia/Karachi",
        },
    )
    values = resp.json()["hourly"]["us_aqi"]
    has_data = any(v is not None for v in values)
    print(f"{date}: {'HAS DATA' if has_data else 'null only'}")