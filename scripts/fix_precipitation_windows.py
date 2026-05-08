from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/precipitation.ipynb")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find block for {label}")
    return text.replace(old, new, 1)


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        if "def fetch_cdo_frame" not in text or "def probe_station_source" not in text:
            continue

        text = replace_once(
            text,
            "PROBE_LIMIT = 25\nAPI_MAX_RETRIES = 5",
            "PROBE_LIMIT = 25\nDOWNLOAD_WINDOW_DAYS = 31\nPROBE_WINDOW_DAYS = 31\nAPI_MAX_RETRIES = 5",
            "precip constants",
        )

        text = replace_once(
            text,
            """def fetch_cdo_frame(token: str | None, dataset_id: str, datatype_id: str, station_id: str, start_date: str, end_date: str, limit: int, kind: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = build_cache_path(kind, dataset_id, datatype_id, station_id, start_date, end_date)
    if cache_path.exists() and not force_refresh:
        frame = pd.read_csv(cache_path, low_memory=False)
        frame.attrs['record_count'] = len(frame)
        frame.attrs['cache_path'] = str(cache_path)
        return frame
    if not token:
        empty = pd.DataFrame()
        empty.attrs['record_count'] = 0
        return empty

    rows = []
    offset = 0
    total_count = None
    while True:
        payload = request_cdo_json(
            'data',
            token,
            params={
                'datasetid': dataset_id,
                'datatypeid': datatype_id,
                'stationid': station_id,
                'startdate': start_date,
                'enddate': end_date,
                'units': 'metric',
                'limit': limit,
                'offset': offset,
            },
        )
        batch = payload.get('results', [])
        if total_count is None:
            total_count = payload.get('metadata', {}).get('resultset', {}).get('count')
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame.to_csv(cache_path, index=False)
    frame.attrs['record_count'] = int(total_count) if total_count is not None else len(frame)
    frame.attrs['cache_path'] = str(cache_path)
    return frame
""",
            """def fetch_cdo_frame(token: str | None, dataset_id: str, datatype_id: str, station_id: str, start_date: str, end_date: str, limit: int, kind: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = build_cache_path(kind, dataset_id, datatype_id, station_id, start_date, end_date)
    if cache_path.exists() and not force_refresh:
        frame = pd.read_csv(cache_path, low_memory=False)
        frame.attrs['record_count'] = len(frame)
        frame.attrs['cache_path'] = str(cache_path)
        return frame
    if not token:
        empty = pd.DataFrame()
        empty.attrs['record_count'] = 0
        return empty

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    window_span = pd.Timedelta(days=DOWNLOAD_WINDOW_DAYS)
    rows = []
    total_count = None
    window_start = start_ts

    while window_start <= end_ts:
        window_end = min(window_start + window_span, end_ts)
        offset = 0

        while True:
            payload = request_cdo_json(
                'data',
                token,
                params={
                    'datasetid': dataset_id,
                    'datatypeid': datatype_id,
                    'stationid': station_id,
                    'startdate': window_start.strftime('%Y-%m-%d'),
                    'enddate': window_end.strftime('%Y-%m-%d'),
                    'units': 'metric',
                    'limit': limit,
                    'offset': offset,
                },
            )
            batch = payload.get('results', [])
            if total_count is None:
                total_count = payload.get('metadata', {}).get('resultset', {}).get('count')
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < limit:
                break
            offset += limit

        window_start = window_end + pd.Timedelta(days=1)
        if window_start <= end_ts:
            time.sleep(0.05)

    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame.to_csv(cache_path, index=False)
    frame.attrs['record_count'] = int(total_count) if total_count is not None else len(frame)
    frame.attrs['cache_path'] = str(cache_path)
    return frame
""",
            "fetch_cdo_frame chunking",
        )

        text = replace_once(
            text,
            """def probe_station_source(token: str | None, dataset_id: str, datatype_id: str, anchor_station_id: str, station_name: str, required_resolution: str, notes: str = '', force_refresh: bool = False) -> dict | None:
    aliases = station_aliases(anchor_station_id, dataset_id=dataset_id)
    cache_key = (dataset_id, datatype_id, anchor_station_id, required_resolution, tuple(aliases))
    if not force_refresh and cache_key in PROBE_CACHE:
        return PROBE_CACHE[cache_key]
    if not token:
        PROBE_CACHE[cache_key] = None
        return None

    errors = []
    for alias in aliases:
        try:
            raw = fetch_cdo_frame(token=token, dataset_id=dataset_id, datatype_id=datatype_id, station_id=alias, start_date=START_DATE, end_date=END_DATE, limit=PROBE_LIMIT, kind='probe', force_refresh=force_refresh)
        except Exception as exc:
            errors.append(f'{alias}: {exc}')
            continue
        if raw.empty:
            continue
        try:
            prepared = prepare_cdo_frame(raw, dataset_id=dataset_id, datatype_id=datatype_id)
        except Exception as exc:
            errors.append(f'{alias}: parse_error={exc}')
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
            """def probe_station_source(token: str | None, dataset_id: str, datatype_id: str, anchor_station_id: str, station_name: str, required_resolution: str, notes: str = '', force_refresh: bool = False) -> dict | None:
    aliases = station_aliases(anchor_station_id, dataset_id=dataset_id)
    cache_key = (dataset_id, datatype_id, anchor_station_id, required_resolution, tuple(aliases))
    if not force_refresh and cache_key in PROBE_CACHE:
        return PROBE_CACHE[cache_key]
    if not token:
        PROBE_CACHE[cache_key] = None
        return None

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
            1,
        )

        cell["source"] = [f"{line}\n" for line in text.rstrip("\n").splitlines()]
        cell["outputs"] = []
        cell["execution_count"] = None
        break

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
