from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from .utils import ensure_directory, fetch_socrata_rows


NYC_311_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "status",
    "borough",
    "incident_zip",
    "incident_address",
    "street_name",
    "latitude",
    "longitude",
]


def month_ranges(start: str, end: str):
    """Yield monthly windows between two ISO dates."""

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    if start_dt >= end_dt:
        raise ValueError("start must be earlier than end")

    current = start_dt
    while current < end_dt:
        next_dt = min(current + relativedelta(months=1), end_dt)
        yield current, next_dt
        current = next_dt


def clean_311_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 311 schema, types, and duplicate keys."""

    if frame.empty:
        return frame.copy()

    cleaned = frame.copy()

    if "created_date" not in cleaned.columns:
        cleaned["created_date"] = pd.NaT
    if "closed_date" not in cleaned.columns:
        cleaned["closed_date"] = pd.NaT
    if "latitude" not in cleaned.columns:
        cleaned["latitude"] = pd.NA
    if "longitude" not in cleaned.columns:
        cleaned["longitude"] = pd.NA

    cleaned["created_date"] = pd.to_datetime(cleaned["created_date"], errors="coerce")
    cleaned["closed_date"] = pd.to_datetime(cleaned["closed_date"], errors="coerce")
    cleaned["latitude"] = pd.to_numeric(cleaned["latitude"], errors="coerce")
    cleaned["longitude"] = pd.to_numeric(cleaned["longitude"], errors="coerce")

    cleaned = cleaned.dropna(subset=["created_date", "latitude", "longitude"])
    if "unique_key" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["unique_key"])
    else:
        cleaned = cleaned.drop_duplicates()

    return cleaned.reset_index(drop=True)


def download_311_dataset(
    dataset_id: str,
    start: str,
    end: str,
    out_dir: str | Path,
    *,
    app_token: str | None = None,
    limit: int = 50_000,
) -> list[Path]:
    """Download one 311 dataset in monthly parquet files."""

    out_path = ensure_directory(Path(out_dir))
    saved_paths: list[Path] = []
    select = ",".join(NYC_311_COLUMNS)

    for window_start, window_end in month_ranges(start, end):
        window_tag = window_start.strftime("%Y_%m")
        parquet_path = out_path / f"{dataset_id}_{window_tag}.parquet"
        if parquet_path.exists():
            saved_paths.append(parquet_path)
            continue

        where = (
            f"created_date >= '{window_start:%Y-%m-%dT%H:%M:%S}.000' "
            f"AND created_date < '{window_end:%Y-%m-%dT%H:%M:%S}.000' "
            "AND latitude IS NOT NULL "
            "AND longitude IS NOT NULL"
        )
        raw = fetch_socrata_rows(
            dataset_id,
            app_token=app_token,
            limit=limit,
            select=select,
            where=where,
            order="created_date, unique_key",
        )
        cleaned = clean_311_frame(raw)
        cleaned.to_parquet(parquet_path, index=False)
        saved_paths.append(parquet_path)

    return saved_paths


def download_default_311_archives(
    out_root: str | Path = Path("data/temporal/311"),
    *,
    app_token: str | None = None,
    limit: int = 50_000,
) -> list[Path]:
    """Download the standard 2010-present and 2020-present 311 archives."""

    today = date.today().isoformat()
    archives = [
        ("erm2-nwe9", "2020-01-01", today, "311_2020_present"),
        ("76ig-c548", "2010-01-01", "2020-01-01", "311_2010_2019"),
    ]

    saved_paths: list[Path] = []
    for dataset_id, start, end, folder in archives:
        saved_paths.extend(
            download_311_dataset(
                dataset_id=dataset_id,
                start=start,
                end=end,
                out_dir=Path(out_root) / folder,
                app_token=app_token,
                limit=limit,
            )
        )

    return saved_paths
