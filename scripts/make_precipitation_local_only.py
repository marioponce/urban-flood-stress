from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/precipitation.ipynb")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find block for {label}")
    return text.replace(old, new, 1)


def replace_block(text: str, start_marker: str, end_marker: str, new_block: str, label: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start == -1 or end == -1:
        raise RuntimeError(f"Could not find block for {label}")
    return text[:start] + new_block + text[end:]


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        if "def fetch_cdo_frame" not in text:
            continue

        text = replace_once(text, "API_MAX_RETRIES = 5", "API_MAX_RETRIES = 10", "retry count")
        text = replace_once(text, "FORCE_REFRESH = True", "FORCE_REFRESH = False", "refresh flag")

        text = replace_block(
            text,
            "def fetch_cdo_frame(",
            "def prepare_cdo_frame(",
            """def fetch_cdo_frame(token: str | None, dataset_id: str, datatype_id: str, station_id: str, start_date: str, end_date: str, limit: int, kind: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = build_cache_path(kind, dataset_id, datatype_id, station_id, start_date, end_date)
    if cache_path.exists():
        frame = pd.read_csv(cache_path, low_memory=False)
        frame.attrs['record_count'] = len(frame)
        frame.attrs['cache_path'] = str(cache_path)
        return frame

    logger.warning(
        'Missing local NOAA cache for kind=%s station=%s dataset=%s datatype=%s start=%s end=%s. Run scripts/download_precipitation_station.py first.',
        kind,
        station_id,
        dataset_id,
        datatype_id,
        start_date,
        end_date,
    )
    empty = pd.DataFrame()
    empty.attrs['record_count'] = 0
    empty.attrs['cache_path'] = str(cache_path)
    return empty

""",
            "local-only fetch",
        )

        text = replace_block(
            text,
            "def discover_precip_datatypes(",
            "def unique_candidates(",
            """def discover_precip_datatypes(dataset_id: str, token: str | None) -> list[dict]:
    if dataset_id in PRECIP_DATATYPE_CACHE:
        return PRECIP_DATATYPE_CACHE[dataset_id]
    PRECIP_DATATYPE_CACHE[dataset_id] = []
    return []

""",
            "local-only datatype discovery",
        )

        text = replace_block(
            text,
            "def hourly_candidate_table(",
            "def probe_station_source(",
            """def hourly_candidate_table(token: str | None) -> list[dict]:
    return unique_candidates(HOURLY_CANDIDATES)

""",
            "static hourly candidates",
        )

        text = replace_block(
            text,
            "def probe_station_source(",
            "def select_precip_source(",
            """def probe_station_source(token: str | None, dataset_id: str, datatype_id: str, anchor_station_id: str, station_name: str, required_resolution: str, notes: str = '', force_refresh: bool = False) -> dict | None:
    aliases = station_aliases(anchor_station_id, dataset_id=dataset_id)
    cache_key = (dataset_id, datatype_id, anchor_station_id, required_resolution, tuple(aliases))
    if not force_refresh and cache_key in PROBE_CACHE:
        return PROBE_CACHE[cache_key]

    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)
    probe_span = pd.Timedelta(days=PROBE_WINDOW_DAYS - 1)
    midpoint = start_ts + (end_ts - start_ts) / 2 if end_ts > start_ts else start_ts
    probe_windows = [
        (start_ts, min(start_ts + probe_span, end_ts)),
        (max(midpoint - probe_span / 2, start_ts), min(midpoint + probe_span / 2, end_ts)),
        (max(end_ts - probe_span, start_ts), end_ts),
    ]
    normalized_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seen_windows: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    for window_start, window_end in probe_windows:
        window_start = pd.Timestamp(window_start).floor('D')
        window_end = pd.Timestamp(window_end).floor('D')
        if window_end < window_start:
            continue
        key = (window_start, window_end)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        normalized_windows.append(key)

    errors = []
    for alias in aliases:
        for window_start, window_end in normalized_windows:
            try:
                raw = fetch_cdo_frame(
                    token=token,
                    dataset_id=dataset_id,
                    datatype_id=datatype_id,
                    station_id=alias,
                    start_date=window_start.strftime('%Y-%m-%d'),
                    end_date=window_end.strftime('%Y-%m-%d'),
                    limit=PROBE_LIMIT,
                    kind='probe',
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                errors.append(f'{alias} {window_start.date()}..{window_end.date()}: {exc}')
                continue
            if raw.empty:
                continue
            try:
                prepared = prepare_cdo_frame(raw, dataset_id=dataset_id, datatype_id=datatype_id)
            except Exception as exc:
                errors.append(f'{alias} {window_start.date()}..{window_end.date()}: parse_error={exc}')
                continue
            if prepared.empty:
                continue
            resolution = infer_temporal_resolution(prepared['timestamp'], dataset_id=dataset_id, datatype_id=datatype_id)
            if required_resolution == 'daily' and resolution != 'daily':
                continue
            if required_resolution == 'hourly' and resolution not in {'hourly', 'subhourly'}:
                continue
            result = {
                'station_id': anchor_station_id,
                'station_name': station_name,
                'source_dataset': dataset_id,
                'data_type': datatype_id,
                'source_station_id': str(prepared['source_station_id'].dropna().iloc[0]) if prepared['source_station_id'].notna().any() else alias,
                'source_alias_used': alias,
                'temporal_resolution': resolution,
                'record_count_probe': int(raw.attrs.get('record_count', len(prepared))),
                'sample_start': prepared['timestamp'].min(),
                'sample_end': prepared['timestamp'].max(),
                'candidate_notes': notes,
            }
            PROBE_CACHE[cache_key] = result
            return result
    if errors:
        logger.info('Probe failed for station=%s dataset=%s datatype=%s examples=%s', anchor_station_id, dataset_id, datatype_id, ' | '.join(errors[:3]))
    PROBE_CACHE[cache_key] = None
    return None

""",
            "local-only probe",
        )

        cell["source"] = [f"{line}\n" for line in text.rstrip("\n").splitlines()]
        cell["outputs"] = []
        cell["execution_count"] = None
        break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        if "token = get_noaa_token()" in text:
            text = replace_once(
                text,
                "token = get_noaa_token()",
                "token = os.getenv('NOAA_CDO_TOKEN') or os.getenv('NOAA_API_TOKEN') or os.getenv('NOAA_TOKEN')",
                "optional token",
            )
            cell["source"] = [f"{line}\n" for line in text.rstrip("\n").splitlines()]
            cell["outputs"] = []
            cell["execution_count"] = None
            break

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
