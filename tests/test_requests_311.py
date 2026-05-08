from pathlib import Path

import pandas as pd

from project_name.requests_311 import (
    clean_311_frame,
    download_311_dataset,
    month_ranges,
)


def test_month_ranges_splits_dates_by_month():
    windows = list(month_ranges("2020-01-15", "2020-03-01"))

    assert len(windows) == 2
    assert windows[0][0].isoformat() == "2020-01-15T00:00:00"
    assert windows[0][1].isoformat() == "2020-02-15T00:00:00"
    assert windows[1][0].isoformat() == "2020-02-15T00:00:00"
    assert windows[1][1].isoformat() == "2020-03-01T00:00:00"


def test_month_ranges_rejects_invalid_interval():
    try:
        list(month_ranges("2020-03-01", "2020-03-01"))
    except ValueError as exc:
        assert "start must be earlier than end" in str(exc)
    else:
        raise AssertionError("month_ranges should reject an empty interval")


def test_clean_311_frame_drops_invalid_rows_and_duplicates():
    raw = pd.DataFrame(
        [
            {
                "unique_key": "1",
                "created_date": "2020-01-01T00:00:00.000",
                "closed_date": "2020-01-01T01:00:00.000",
                "latitude": "40.0",
                "longitude": "-73.0",
            },
            {
                "unique_key": "1",
                "created_date": "2020-01-01T00:00:00.000",
                "closed_date": "2020-01-01T01:00:00.000",
                "latitude": "40.0",
                "longitude": "-73.0",
            },
            {
                "unique_key": "2",
                "created_date": "bad-date",
                "closed_date": None,
                "latitude": "40.1",
                "longitude": "-73.1",
            },
            {
                "unique_key": "3",
                "created_date": "2020-01-02T00:00:00.000",
                "closed_date": None,
                "latitude": None,
                "longitude": "-73.2",
            },
        ]
    )

    cleaned = clean_311_frame(raw)

    assert list(cleaned["unique_key"]) == ["1"]
    assert pd.api.types.is_datetime64_any_dtype(cleaned["created_date"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["closed_date"])
    assert pd.api.types.is_float_dtype(cleaned["latitude"])
    assert pd.api.types.is_float_dtype(cleaned["longitude"])


def test_download_311_dataset_writes_cleaned_monthly_output(monkeypatch, tmp_path):
    calls = []

    class FakeSocrata:
        def __init__(self, domain, app_token, timeout):
            assert domain == "data.cityofnewyork.us"
            assert app_token == "token"
            assert timeout == 120

        def get(self, dataset_id, **kwargs):
            calls.append({"dataset_id": dataset_id, **kwargs})

            if kwargs["offset"] > 0:
                return []

            return [
                {
                    "unique_key": "1",
                    "created_date": "2020-01-01T00:00:00.000",
                    "closed_date": "2020-01-01T01:00:00.000",
                    "agency": "DEP",
                    "agency_name": "Department of Environmental Protection",
                    "complaint_type": "Flooding",
                    "descriptor": "Street Flooding",
                    "status": "Closed",
                    "borough": "BROOKLYN",
                    "incident_zip": "11201",
                    "incident_address": "1 Main St",
                    "street_name": "Main St",
                    "latitude": "40.0",
                    "longitude": "-73.0",
                },
                {
                    "unique_key": "1",
                    "created_date": "2020-01-01T00:00:00.000",
                    "closed_date": "2020-01-01T01:00:00.000",
                    "agency": "DEP",
                    "agency_name": "Department of Environmental Protection",
                    "complaint_type": "Flooding",
                    "descriptor": "Street Flooding",
                    "status": "Closed",
                    "borough": "BROOKLYN",
                    "incident_zip": "11201",
                    "incident_address": "1 Main St",
                    "street_name": "Main St",
                    "latitude": "40.0",
                    "longitude": "-73.0",
                },
            ]

    written = {}

    def fake_to_parquet(self, path, index=False):
        written["path"] = Path(path)
        written["index"] = index
        written["frame"] = self.copy()

    monkeypatch.setattr("project_name.utils.Socrata", FakeSocrata)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    download_311_dataset(
        dataset_id="erm2-nwe9",
        start="2020-01-01",
        end="2020-02-01",
        out_dir=str(tmp_path),
        app_token="token",
        limit=100,
    )

    assert len(calls) == 2
    assert calls[0]["dataset_id"] == "erm2-nwe9"
    assert calls[0]["order"] == "created_date, unique_key"
    assert calls[0]["limit"] == 100
    assert calls[0]["offset"] == 0
    assert calls[1]["offset"] == 100
    assert written["path"].name == "erm2-nwe9_2020_01.parquet"
    assert written["index"] is False
    assert len(written["frame"]) == 1
