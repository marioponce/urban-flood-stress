from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE_DIR = ROOT / "data" / "temporal" / "noaa" / "cache"
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

NOAA_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
NOAA_TOKEN_SLOT_ENV_NAMES = (
    "NOAA_CDO_TOKEN_0",
    "NOAA_CDO_TOKEN_1",
    "NOAA_CDO_TOKEN_2",
    "NOAA_CDO_TOKEN_3",
)
NOAA_TOKEN_ENV_NAMES = ("NOAA_CDO_TOKEN", "NOAA_API_TOKEN", "NOAA_TOKEN")
START_DATE = "2010-01-01"
END_DATE = date.today().isoformat()
WINDOW_DAYS = 31
PROBE_WINDOW_DAYS = 31
API_LIMIT = 1000
MAX_RETRIES = 6
BACKOFF_SECONDS = 2.0
REQUEST_MIN_INTERVAL_SECONDS = 0.30
TOKEN_STATE_LOCK = threading.Lock()
TOKEN_STATES: dict[str, dict[str, object]] = {}

TARGET_STATIONS = [
    "US1NJPS0012",
    "USC00283704",
    "USC00289187",
    "US1NJMN0010",
    "USC00066655",
    "US1NJMS0049",
    "US1NJPS0019",
    "USC00300961",
    "USC00287865",
    "USC00285503",
    "USC00302129",
    "USC00287079",
    "USC00306138",
]

DAILY_CANDIDATES = [
    {"dataset_id": "GHCND", "datatype_id": "PRCP", "expected_resolution": "daily", "notes": "GHCND daily precipitation"},
]

HOURLY_CANDIDATES = [
    {"dataset_id": "PRECIP_HLY", "datatype_id": "HPCP", "expected_resolution": "hourly", "notes": "Hourly Precipitation Data / HPCP"},
    {"dataset_id": "LCD", "datatype_id": "HourlyPrecipitation", "expected_resolution": "hourly", "notes": "Local Climatological Data hourly precipitation"},
    {"dataset_id": "LCD", "datatype_id": "HPCP", "expected_resolution": "hourly", "notes": "LCD HPCP fallback when available"},
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ROOT / ".env")


def build_cache_path(kind: str, dataset_id: str, datatype_id: str, station_id: str, start_date: str, end_date: str) -> Path:
    import hashlib

    key = f"{kind}|{dataset_id}|{datatype_id}|{station_id}|{start_date}|{end_date}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    safe_dataset = dataset_id.replace(":", "_")
    safe_datatype = datatype_id.replace(":", "_")
    return RAW_CACHE_DIR / f"precip_{kind}_{safe_dataset}_{safe_datatype}_{digest}.csv"


def station_aliases(station_code: str, dataset_id: str | None = None) -> list[str]:
    aliases = [station_code]
    if dataset_id == "GHCND":
        aliases.insert(0, f"GHCND:{station_code}")
    elif dataset_id == "PRECIP_HLY":
        aliases = [f"COOP:{station_code}", f"GHCND:{station_code}", station_code]
    elif dataset_id == "LCD":
        aliases = [f"LCD:{station_code}", f"COOP:{station_code}", f"GHCND:{station_code}", station_code]
    elif dataset_id == "GLOBAL_HOURLY":
        aliases = [f"GHCND:{station_code}", f"COOP:{station_code}", station_code]
    aliases.extend([f"COOP:{station_code}", f"GHCND:{station_code}", f"LCD:{station_code}"])
    out: list[str] = []
    seen = set()
    for alias in aliases:
        if alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


def load_noaa_tokens() -> list[tuple[str, str]]:
    """Load NOAA tokens from .env or the environment.

    Prefer the numbered token slots so we can run requests in parallel without
    sharing one rate limit bucket.
    """

    tokens: list[tuple[str, str]] = []
    for name in NOAA_TOKEN_SLOT_ENV_NAMES:
        value = os.getenv(name)
        if value:
            tokens.append((name, value))
    if tokens:
        return tokens

    for name in NOAA_TOKEN_ENV_NAMES:
        value = os.getenv(name)
        if value:
            tokens.append((name, value))
    if tokens:
        return tokens

    raise RuntimeError(
        "Missing NOAA token. Define NOAA_CDO_TOKEN_0..NOAA_CDO_TOKEN_3 or NOAA_CDO_TOKEN, NOAA_API_TOKEN, NOAA_TOKEN."
    )


def get_token_state(token: str) -> dict[str, object]:
    with TOKEN_STATE_LOCK:
        state = TOKEN_STATES.get(token)
        if state is None:
            state = {
                "session": requests.Session(),
                "last_request_at": 0.0,
                "lock": threading.Lock(),
            }
            TOKEN_STATES[token] = state
        return state


def request_cdo_json(endpoint: str, token: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    url = f"{NOAA_BASE_URL}/{endpoint}"
    headers = {"token": token}
    state = get_token_state(token)
    session = state["session"]
    last_error: Exception | None = None

    def throttle() -> None:
        lock = state["lock"]
        with lock:
            now = time.monotonic()
            elapsed = now - float(state["last_request_at"])
            if elapsed < REQUEST_MIN_INTERVAL_SECONDS:
                time.sleep(REQUEST_MIN_INTERVAL_SECONDS - elapsed)
            state["last_request_at"] = time.monotonic()

    for attempt in range(1, retries + 1):
        try:
            throttle()
            response = session.get(url, headers=headers, params=params, timeout=120)
        except Exception as exc:
            last_error = exc
            sleep_for = BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"[WARNING] NOAA retry endpoint={endpoint} attempt={attempt}/{retries} sleep={sleep_for:.1f}s exc={exc}")
            time.sleep(sleep_for)
            continue

        if response.status_code in {429, 500, 502, 503, 504}:
            sleep_for = BACKOFF_SECONDS * (2 ** (attempt - 1))
            body = response.text.strip().replace("\n", " ")[:250]
            print(f"[WARNING] stationid={params.get('stationid')} NOAA retry status={response.status_code} endpoint={endpoint} attempt={attempt}/{retries} sleep={sleep_for:.1f}s body={body}")
            last_error = RuntimeError(body or f"HTTP {response.status_code}")
            time.sleep(sleep_for)
            continue
        elif response.ok:
            print(f"[INFO] NOAA success dataset_id={params.get('datasetid')} station_id={params.get('stationid')}")

        try:
            response.raise_for_status()
        except Exception as exc:
            body = response.text.strip().replace("\n", " ")[:500]
            raise RuntimeError(f"Request failed for {url}: {response.status_code} {response.reason}. {body}") from exc

        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"NOAA error for {url}: {payload['error']}")
        return payload

    if last_error is not None:
        raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error
    raise RuntimeError(f"Unable to fetch {url}")


def fetch_window_rows(
    token: str,
    dataset_id: str,
    datatype_id: str,
    station_id: str,
    start_date: str,
    end_date: str,
    limit: int = API_LIMIT,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    offset = 0
    total_count = 0

    while True:
        payload = request_cdo_json(
            "data",
            token,
            params={
                "datasetid": dataset_id,
                "datatypeid": datatype_id,
                "stationid": station_id,
                "startdate": start_date,
                "enddate": end_date,
                "units": "metric",
                "limit": limit,
                "offset": offset,
            },
        )
        batch = payload.get("results", []) or []
        resultset = payload.get("metadata", {}).get("resultset", {})
        if resultset.get("count") is not None:
            total_count = int(resultset["count"])
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    frame = pd.DataFrame.from_records(rows)
    return frame, total_count or len(frame)


def prepare_cdo_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "source_station_id", "precip_mm"])
    rename_map = {}
    if "date" in out.columns and "timestamp" not in out.columns:
        rename_map["date"] = "timestamp"
    if "station" in out.columns and "source_station_id" not in out.columns:
        rename_map["station"] = "source_station_id"
    if "value" in out.columns and "precip_mm" not in out.columns:
        rename_map["value"] = "precip_mm"
    out = out.rename(columns=rename_map)
    if "timestamp" not in out.columns or "precip_mm" not in out.columns:
        raise ValueError(f"NOAA response lacks required columns: {sorted(out.columns)}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    raw = out["precip_mm"].astype("string").str.strip()
    trace_mask = raw.str.upper().isin({"T", "TRACE"})
    raw = raw.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    out["precip_mm"] = pd.to_numeric(raw, errors="coerce")
    out.loc[trace_mask, "precip_mm"] = 0.0
    out = out.dropna(subset=["timestamp", "precip_mm"]).copy()
    out = out.loc[out["precip_mm"] >= 0].copy()
    if "source_station_id" not in out.columns:
        out["source_station_id"] = pd.NA
    out["source_station_id"] = out["source_station_id"].astype("string")
    return out[["timestamp", "source_station_id", "precip_mm"]].copy()


def infer_temporal_resolution(timestamps) -> str:
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values().drop_duplicates()
    if len(ts) < 2:
        return "unknown"
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return "unknown"
    min_delta = diffs.min()
    if min_delta < pd.Timedelta(hours=1):
        return "subhourly"
    if min_delta < pd.Timedelta(days=1):
        return "hourly"
    return "daily"


def window_ranges(start_date: str, end_date: str, window_days: int = WINDOW_DAYS):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    window_start = start_ts
    span = pd.Timedelta(days=window_days)
    while window_start <= end_ts:
        window_end = min(window_start + span, end_ts)
        yield window_start, window_end
        window_start = window_end + pd.Timedelta(days=1)


def probe_windows(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    span = pd.Timedelta(days=PROBE_WINDOW_DAYS - 1)
    midpoint = start_ts + (end_ts - start_ts) / 2 if end_ts > start_ts else start_ts
    raw_windows = [
        (start_ts, min(start_ts + span, end_ts)),
        (max(midpoint - span / 2, start_ts), min(midpoint + span / 2, end_ts)),
        (max(end_ts - span, start_ts), end_ts),
    ]
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seen = set()
    for window_start, window_end in raw_windows:
        window_start = pd.Timestamp(window_start).floor("D")
        window_end = pd.Timestamp(window_end).floor("D")
        if window_end < window_start:
            continue
        key = (window_start, window_end)
        if key in seen:
            continue
        windows.append(key)
        seen.add(key)
    return windows


def save_frame(frame: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)


def download_frame_for_window(
    token: str,
    dataset_id: str,
    datatype_id: str,
    station_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for window_start, window_end in window_ranges(start_date, end_date):
        frame, _ = fetch_window_rows(
            token=token,
            dataset_id=dataset_id,
            datatype_id=datatype_id,
            station_id=station_id,
            start_date=window_start.strftime("%Y-%m-%d"),
            end_date=window_end.strftime("%Y-%m-%d"),
        )
        if not frame.empty:
            pieces.append(frame)
        if window_end < pd.Timestamp(end_date):
            time.sleep(0.05)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def probe_candidate(
    token: str,
    station_id: str,
    dataset_id: str,
    datatype_id: str,
    required_resolution: str,
) -> dict | None:
    aliases = station_aliases(station_id, dataset_id=dataset_id)
    windows = probe_windows(START_DATE, END_DATE)

    for alias in aliases:
        for window_start, window_end in windows:
            probe_cache = build_cache_path("probe", dataset_id, datatype_id, alias, window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))
            if probe_cache.exists():
                raw = pd.read_csv(probe_cache, low_memory=False)
            else:
                try:
                    raw, _ = fetch_window_rows(
                        token=token,
                        dataset_id=dataset_id,
                        datatype_id=datatype_id,
                        station_id=alias,
                        start_date=window_start.strftime("%Y-%m-%d"),
                        end_date=window_end.strftime("%Y-%m-%d"),
                    )
                except Exception as exc:
                    print(f"[INFO] Probe failed for station={station_id} dataset={dataset_id} datatype={datatype_id} alias={alias} window={window_start.date()}..{window_end.date()}: {exc}")
                    continue
                if not raw.empty:
                    save_frame(raw, probe_cache)

            if raw.empty:
                continue

            prepared = prepare_cdo_frame(raw)
            if prepared.empty:
                continue
            resolution = infer_temporal_resolution(prepared["timestamp"])
            if required_resolution == "daily" and resolution != "daily":
                continue
            if required_resolution == "hourly" and resolution not in {"hourly", "subhourly"}:
                continue
            return {
                "station_id": station_id,
                "source_station_id": alias,
                "source_dataset": dataset_id,
                "data_type": datatype_id,
                "temporal_resolution": resolution,
            }
    return None


def download_selected_source(token: str, probe: dict, kind: str, force: bool = False) -> Path:
    cache_path = build_cache_path(
        kind,
        probe["source_dataset"],
        probe["data_type"],
        probe["source_station_id"],
        START_DATE,
        END_DATE,
    )
    if cache_path.exists() and not force:
        return cache_path
    frames: list[pd.DataFrame] = []
    for window_start, window_end in window_ranges(START_DATE, END_DATE):
        frame, _ = fetch_window_rows(
            token=token,
            dataset_id=probe["source_dataset"],
            datatype_id=probe["data_type"],
            station_id=probe["source_station_id"],
            start_date=window_start.strftime("%Y-%m-%d"),
            end_date=window_end.strftime("%Y-%m-%d"),
        )
        if not frame.empty:
            frames.append(frame)
        if window_end < pd.Timestamp(END_DATE):
            time.sleep(0.05)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    save_frame(combined, cache_path)
    return cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA precipitation data for one station into local caches.")
    parser.add_argument("--station-index", type=int, help="1-based index into the target station list.")
    parser.add_argument("--station-indexes", type=int, nargs="+", help="One or more 1-based station indexes to download sequentially.")
    parser.add_argument("--station-id", type=str, help="Explicit NOAA station id to download.")
    parser.add_argument("--all-stations", action="store_true", help="Download every station in the fixed list sequentially.")
    parser.add_argument("--workers", type=int, default=None, help="Maximum number of stations to process in parallel. Defaults to the number of loaded NOAA tokens.")
    parser.add_argument("--force", action="store_true", help="Redownload even if caches already exist.")
    parser.add_argument("--kind", choices=["daily", "hourly", "both"], default="both", help="What to download.")
    parser.add_argument("--list-stations", action="store_true", help="Print the fixed station mapping and exit.")
    return parser.parse_args()


def resolve_station_ids(args: argparse.Namespace) -> list[str]:
    if args.all_stations:
        return TARGET_STATIONS.copy()
    if args.station_indexes:
        ids = []
        for index in args.station_indexes:
            if index < 1 or index > len(TARGET_STATIONS):
                raise SystemExit(f"--station-indexes must be between 1 and {len(TARGET_STATIONS)}")
            ids.append(TARGET_STATIONS[index - 1])
        return ids
    if args.station_id:
        return [args.station_id]
    if args.station_index is None:
        raise SystemExit("Provide --station-index or --station-id.")
    if args.station_index < 1 or args.station_index > len(TARGET_STATIONS):
        raise SystemExit(f"--station-index must be between 1 and {len(TARGET_STATIONS)}")
    return [TARGET_STATIONS[args.station_index - 1]]


def process_station_download(
    station_id: str,
    token_name: str,
    token: str,
    kind: str,
    force: bool = False,
) -> dict:
    print(f"[INFO] Station {station_id} -> token {token_name}")
    selected: list[tuple[str, dict]] = []

    try:
        if kind in {"daily", "both"}:
            for candidate in DAILY_CANDIDATES:
                probe = probe_candidate(token, station_id, candidate["dataset_id"], candidate["datatype_id"], "daily")
                if probe is not None:
                    selected.append(("daily", probe))
                    break
        if kind in {"hourly", "both"}:
            for candidate in HOURLY_CANDIDATES:
                probe = probe_candidate(token, station_id, candidate["dataset_id"], candidate["datatype_id"], "hourly")
                if probe is not None:
                    selected.append(("hourly", probe))
                    break

        if not selected:
            print(f"[WARNING] No precipitation source found for {station_id}")
            return {"station_id": station_id, "token_name": token_name, "ok": False, "reason": "no_source"}

        cache_paths: list[str] = []
        for kind_name, probe in selected:
            cache_path = download_selected_source(token, probe, kind_name, force=force)
            cache_paths.append(str(cache_path))
            print(f"[INFO] Cached {kind_name} -> {cache_path}")

        return {
            "station_id": station_id,
            "token_name": token_name,
            "ok": True,
            "selected": selected,
            "cache_paths": cache_paths,
        }
    except Exception as exc:
        print(f"[ERROR] Station {station_id} failed on {token_name}: {exc}")
        return {"station_id": station_id, "token_name": token_name, "ok": False, "reason": str(exc)}


def main() -> None:
    args = parse_args()
    if args.list_stations:
        for i, station_id in enumerate(TARGET_STATIONS, start=1):
            print(f"{i:02d} -> {station_id}")
        return

    token_entries = load_noaa_tokens()
    station_ids = resolve_station_ids(args)
    worker_limit = args.workers if args.workers is not None else len(token_entries)
    worker_limit = max(1, min(worker_limit, len(token_entries)))
    print(f"[INFO] Loaded {len(token_entries)} NOAA token(s); running up to {worker_limit} station(s) in parallel.")

    results: list[dict] = []
    batch_number = 0
    for start_idx in range(0, len(station_ids), worker_limit):
        batch_number += 1
        batch_station_ids = station_ids[start_idx : start_idx + worker_limit]
        batch_tokens = token_entries[: len(batch_station_ids)]
        print(f"[INFO] Batch {batch_number}: {len(batch_station_ids)} station(s)")

        with ThreadPoolExecutor(max_workers=len(batch_station_ids)) as executor:
            futures = [
                executor.submit(
                    process_station_download,
                    station_id,
                    token_name,
                    token,
                    args.kind,
                    args.force,
                )
                for station_id, (token_name, token) in zip(batch_station_ids, batch_tokens)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        if start_idx + worker_limit < len(station_ids):
            time.sleep(1.0)

    successes = sum(1 for result in results if result.get("ok"))
    print(f"[INFO] Done. {successes}/{len(results)} station job(s) succeeded.")


if __name__ == "__main__":
    main()
