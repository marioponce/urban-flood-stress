from __future__ import annotations

from pathlib import Path

import pandas as pd
from sodapy import Socrata


DEFAULT_SOCRATA_DOMAIN = "data.cityofnewyork.us"


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_socrata_rows(
    dataset_id: str,
    *,
    app_token: str | None = None,
    domain: str = DEFAULT_SOCRATA_DOMAIN,
    limit: int = 50_000,
    timeout: int = 120,
    select: str | None = None,
    where: str | None = None,
    order: str | None = None,
) -> pd.DataFrame:
    """Fetch a Socrata dataset page by page and return a DataFrame."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    client = Socrata(domain, app_token, timeout=timeout)
    rows: list[dict] = []
    offset = 0

    while True:
        query: dict[str, object] = {"limit": limit, "offset": offset}
        if select is not None:
            query["select"] = select
        if where is not None:
            query["where"] = where
        if order is not None:
            query["order"] = order

        batch = client.get(dataset_id, **query)
        if not batch:
            break

        rows.extend(batch)
        offset += limit

    return pd.DataFrame.from_records(rows)
