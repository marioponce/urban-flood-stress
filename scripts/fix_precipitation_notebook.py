from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/precipitation.ipynb")


def cell_lines(text: str) -> list[str]:
    text = dedent(text).strip("\n")
    if not text:
        return []
    return [f"{line}\n" for line in text.splitlines()]


def set_cell(nb: dict, index: int, text: str) -> None:
    cell = nb["cells"][index]
    cell["source"] = cell_lines(text)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())

    set_cell(
        nb,
        0,
        """
        # Precipitation

        Descarga el registro completo de precipitacion desde NOAA CDO para las estaciones seleccionadas, construye Thiessen recortado a NYC y segmenta eventos de lluvia por estacion.

        Regla de eventos:
        - separacion minima: `6 h` sin lluvia;
        - umbral minimo del evento: `1.1 mm`;
        - limpieza previa: valores negativos fuera.

        Como el registro disponible para estas estaciones es diario, la separacion efectiva se adapta al intervalo observado para no fragmentar todos los eventos en bloques de un solo dia.

        La ultima celda muestra como filtrar las estaciones que quieres analizar.
        """,
    )

    set_cell(
        nb,
        1,
        """
        import hashlib
        import os
        import time
        from datetime import date, timedelta
        from pathlib import Path

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import requests
        from shapely import make_valid, voronoi_polygons
        from shapely.ops import unary_union

        MPLCONFIGDIR = Path('/tmp/matplotlib')
        MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIGDIR))

        plt.style.use('seaborn-v0_8-whitegrid')

        def find_project_root() -> Path:
            # Locate the repository root from any notebook working directory.
            current = Path.cwd().resolve()
            for candidate in (current, *current.parents):
                if (candidate / 'pyproject.toml').exists() and (candidate / 'data').exists():
                    return candidate
            return current


        ROOT = find_project_root()

        INVENTORY_PATH = ROOT / 'data' / 'temporal' / 'noaa' / '4308211.csv'
        SPATIAL_OUTPUT_DIR = ROOT / 'data' / 'spatial' / 'vector' / 'noaa'
        TEMPORAL_OUTPUT_DIR = ROOT / 'data' / 'temporal' / 'noaa'
        RAW_CACHE_DIR = TEMPORAL_OUTPUT_DIR / 'cache'
        EVENT_DIR = TEMPORAL_OUTPUT_DIR / 'precip_events'
        EVENT_SUMMARY_PATH = TEMPORAL_OUTPUT_DIR / 'precipitation_event_summary.csv'

        for folder in (SPATIAL_OUTPUT_DIR, TEMPORAL_OUTPUT_DIR, RAW_CACHE_DIR, EVENT_DIR):
            folder.mkdir(parents=True, exist_ok=True)

        NOAA_BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
        NOAA_TOKEN_ENV_NAMES = ('NOAA_CDO_TOKEN', 'NOAA_API_TOKEN')
        NOAA_DATASET_ID = 'GHCND'
        NOAA_DATATYPE_ID = 'PRCP'
        NOAA_START_DATE = '2010-01-01'
        NOAA_END_DATE = date.today().isoformat()
        NOAA_LIMIT = 1000
        NOAA_MAX_RETRIES = 5
        NOAA_RETRY_BACKOFF_SECONDS = 2.0

        EVENT_GAP = pd.Timedelta(hours=6)
        MIN_EVENT_MM = 1.1
        """,
    )

    set_cell(
        nb,
        2,
        """
        def load_nyc_boundary():
            # Load the NYC borough polygons and a valid union geometry.
            nyc = gpd.read_file(ROOT / 'data' / 'spatial' / 'vector' / 'nyc_borough_boundary' / 'nybb.geojson')
            nyc = nyc.copy()
            nyc['geometry'] = nyc.geometry.apply(make_valid)
            city = unary_union(nyc.geometry)
            return nyc, city


        def build_thiessen(points_gdf, clip_geom):
            # Build Thiessen polygons, clipped to the NYC union geometry.
            if len(points_gdf) < 2:
                raise ValueError('Se necesitan al menos dos estaciones para construir Thiessen.')
            diagram = voronoi_polygons(
                unary_union(points_gdf.geometry),
                extend_to=clip_geom,
                ordered=True,
            )
            rows = []
            base_rows = points_gdf.drop(columns='geometry').to_dict('records')
            for record, cell in zip(base_rows, diagram.geoms):
                clipped = cell.intersection(clip_geom)
                if clipped.is_empty:
                    continue
                record = dict(record)
                record['geometry'] = clipped
                rows.append(record)
            return gpd.GeoDataFrame(rows, crs=points_gdf.crs)


        def load_station_inventory():
            # Load the station catalog with coordinates and NOAA ids.
            inventory = pd.read_csv(
                INVENTORY_PATH,
                usecols=['STATION', 'NAME', 'LATITUDE', 'LONGITUDE'],
                dtype={'STATION': 'string', 'NAME': 'string'},
            ).drop_duplicates('STATION')
            inventory = inventory.rename(
                columns={
                    'STATION': 'station_code',
                    'NAME': 'station_name',
                    'LATITUDE': 'lat',
                    'LONGITUDE': 'lon',
                }
            )
            inventory['station_id'] = 'GHCND:' + inventory['station_code'].astype(str)
            inventory['role'] = 'inventory'
            return inventory.sort_values('station_id').reset_index(drop=True)


        def infer_observation_step(timestamps):
            # Infer the sampling interval from the observed timestamps.
            clean = pd.Series(pd.to_datetime(timestamps, errors='coerce')).dropna().sort_values().drop_duplicates()
            if len(clean) < 2:
                return pd.Timedelta(days=1)
            diffs = clean.diff().dropna()
            diffs = diffs[diffs > pd.Timedelta(0)]
            if diffs.empty:
                return pd.Timedelta(days=1)
            mode = diffs.mode()
            if not mode.empty:
                return mode.iloc[0]
            return diffs.median()


        def precip_cache_path(station_ids, dataset_id=NOAA_DATASET_ID, datatype_id=NOAA_DATATYPE_ID,
                              start_date=NOAA_START_DATE, end_date=NOAA_END_DATE):
            # Build a stable cache path for a station selection and date range.
            clean_ids = [str(station_id) for station_id in station_ids if pd.notna(station_id)]
            clean_ids = sorted(set(clean_ids))
            if not clean_ids:
                raise ValueError('No hay estaciones seleccionadas para la descarga NOAA.')
            selection_key = '|'.join(clean_ids)
            selection_hash = hashlib.sha1(selection_key.encode('utf-8')).hexdigest()[:12]
            name = f'noaa_precip_raw_{dataset_id}_{datatype_id}_{start_date}_{end_date}_{selection_hash}.csv'
            return RAW_CACHE_DIR / name


        def normalize_precip_frame(frame):
            # Normalize NOAA precipitation records to the notebook schema.
            out = frame.copy()
            rename_map = {}
            if 'date' in out.columns and 'timestamp' not in out.columns:
                rename_map['date'] = 'timestamp'
            if 'station' in out.columns and 'station_id' not in out.columns:
                rename_map['station'] = 'station_id'
            if 'value' in out.columns and 'precip_mm' not in out.columns:
                rename_map['value'] = 'precip_mm'
            out = out.rename(columns=rename_map)

            required = {'timestamp', 'station_id', 'precip_mm'}
            missing = required - set(out.columns)
            if missing:
                raise ValueError(f'La descarga NOAA no tiene las columnas esperadas: {sorted(missing)}')

            out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
            out['station_id'] = out['station_id'].astype('string')
            out['precip_mm'] = pd.to_numeric(out['precip_mm'], errors='coerce')
            out = out.dropna(subset=['timestamp', 'station_id', 'precip_mm'])
            out = out.loc[out['precip_mm'] >= 0].copy()
            out = out.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
            out['source_kind'] = 'noaa_cdo_api'
            return out


        def fetch_noaa_data(
            token: str,
            dataset_id: str,
            datatype_id: str,
            start_date: str,
            end_date: str,
            stations: list[str],
            limit: int,
            max_retries: int,
            retry_backoff_seconds: float,
        ) -> pd.DataFrame:
            # Fetch NOAA CDO data in windows so long station histories remain manageable.
            headers = {'token': token}
            rows = []
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            window_start = start_ts

            while window_start <= end_ts:
                window_end = min(window_start + timedelta(days=364), end_ts)
                offset = 1

                while True:
                    params = {
                        'datasetid': dataset_id,
                        'datatypeid': datatype_id,
                        'startdate': window_start.strftime('%Y-%m-%d'),
                        'enddate': window_end.strftime('%Y-%m-%d'),
                        'limit': limit,
                        'offset': offset,
                        'units': 'metric',
                    }
                    for station in stations:
                        params.setdefault('stationid', [])
                        params['stationid'].append(station)

                    response = None
                    for attempt in range(max_retries + 1):
                        response = requests.get(
                            NOAA_BASE_URL,
                            headers=headers,
                            params=params,
                            timeout=60,
                        )
                        if response.status_code not in {429, 500, 502, 503, 504}:
                            break
                        if attempt == max_retries:
                            break
                        wait_seconds = retry_backoff_seconds * (2 ** attempt)
                        print(
                            '[retry] NOAA request failed '
                            f"status={response.status_code} "
                            f"window={params['startdate']}..{params['enddate']} "
                            f'offset={offset} '
                            f'attempt={attempt + 1}/{max_retries + 1} '
                            f'sleep={wait_seconds:.1f}s'
                        )
                        time.sleep(wait_seconds)

                    response.raise_for_status()
                    payload = response.json()
                    batch = payload.get('results', [])
                    if not batch:
                        break

                    rows.extend(batch)
                    offset += limit
                    if len(batch) < limit:
                        break

                window_start = window_end + timedelta(days=1)

            return pd.DataFrame.from_records(rows)


        def load_or_fetch_precip(station_ids, force_refresh=False):
            # Load a cached NOAA precipitation extract or fetch it from the API.
            cache_path = precip_cache_path(station_ids)
            if cache_path.exists() and not force_refresh:
                frame = pd.read_csv(cache_path, low_memory=False)
                return normalize_precip_frame(frame)

            token = next((os.getenv(name) for name in NOAA_TOKEN_ENV_NAMES if os.getenv(name)), None)
            if not token:
                raise RuntimeError(
                    'Falta el token de NOAA. Define NOAA_CDO_TOKEN o NOAA_API_TOKEN para descargar el registro completo.'
                )

            raw = fetch_noaa_data(
                token=token,
                dataset_id=NOAA_DATASET_ID,
                datatype_id=NOAA_DATATYPE_ID,
                start_date=NOAA_START_DATE,
                end_date=NOAA_END_DATE,
                stations=[str(station_id) for station_id in station_ids],
                limit=NOAA_LIMIT,
                max_retries=NOAA_MAX_RETRIES,
                retry_backoff_seconds=NOAA_RETRY_BACKOFF_SECONDS,
            )
            frame = normalize_precip_frame(raw)
            frame.to_csv(cache_path, index=False)
            return frame


        def set_map_extent(ax, geom, pad=0.03):
            # Set a small padded view around a geometry.
            minx, miny, maxx, maxy = geom.bounds
            dx = (maxx - minx) * pad
            dy = (maxy - miny) * pad
            ax.set_xlim(minx - dx, maxx + dx)
            ax.set_ylim(miny - dy, maxy + dy)


        def split_rain_events(station_frame, station_id, station_name, gap=EVENT_GAP, min_total=MIN_EVENT_MM):
            # Split one station series into independent rain events and write a CSV per event.
            frame = station_frame.copy().sort_values('timestamp').reset_index(drop=True)
            frame = frame.dropna(subset=['timestamp', 'precip_mm'])
            frame = frame.loc[frame['precip_mm'] >= 0].copy()

            if frame.empty:
                return pd.DataFrame(
                    columns=[
                        'station_id', 'station_name', 'event_id', 'event_uid', 'start', 'end',
                        'duration_hours', 'precip_mm', 'mean_intensity_mm_per_hr', 'n_records',
                        'wet_records', 'csv_path',
                    ]
                )

            sample_step = infer_observation_step(frame['timestamp'])
            effective_gap = max(gap, sample_step * 1.5)

            wet = frame.loc[frame['precip_mm'] > 0, ['timestamp', 'precip_mm']].copy()
            if wet.empty:
                return pd.DataFrame(
                    columns=[
                        'station_id', 'station_name', 'event_id', 'event_uid', 'start', 'end',
                        'duration_hours', 'precip_mm', 'mean_intensity_mm_per_hr', 'n_records',
                        'wet_records', 'csv_path',
                    ]
                )

            wet['time_diff'] = wet['timestamp'].diff()
            wet['is_new_event'] = wet['time_diff'].isna() | (wet['time_diff'] > effective_gap)
            wet['event_id'] = wet['is_new_event'].cumsum().astype(int)

            summaries = []
            station_slug = str(station_id).replace(':', '_')
            station_dir = EVENT_DIR / station_slug
            station_dir.mkdir(parents=True, exist_ok=True)

            for event_id, wet_group in wet.groupby('event_id', sort=True):
                wet_group = wet_group.sort_values('timestamp')
                start = wet_group['timestamp'].min()
                end = wet_group['timestamp'].max()
                event_frame = frame.loc[(frame['timestamp'] >= start) & (frame['timestamp'] <= end)].copy()
                if event_frame.empty:
                    continue

                total_mm = float(event_frame['precip_mm'].sum())
                if total_mm < min_total:
                    continue

                duration_hours = float(max(((end - start) + sample_step).total_seconds() / 3600.0, sample_step.total_seconds() / 3600.0))
                intensity = total_mm / duration_hours if duration_hours > 0 else np.nan
                event_uid = f'{station_slug}_{int(event_id):04d}'
                event_path = station_dir / f'event_{int(event_id):04d}.csv'

                event_frame = event_frame.assign(
                    event_id=int(event_id),
                    event_uid=event_uid,
                    station_id=station_id,
                    station_name=station_name,
                )
                event_frame.to_csv(event_path, index=False)

                summaries.append(
                    {
                        'station_id': station_id,
                        'station_name': station_name,
                        'event_id': int(event_id),
                        'event_uid': event_uid,
                        'start': start,
                        'end': end,
                        'duration_hours': duration_hours,
                        'precip_mm': total_mm,
                        'mean_intensity_mm_per_hr': intensity,
                        'n_records': int(len(event_frame)),
                        'wet_records': int(len(wet_group)),
                        'csv_path': str(event_path),
                    }
                )

            return pd.DataFrame(summaries)
        """,
    )

    set_cell(
        nb,
        3,
        """
        nyc_boroughs, nyc_union = load_nyc_boundary()
        station_catalog = load_station_inventory()

        seed_points = gpd.GeoDataFrame(
            station_catalog,
            geometry=gpd.points_from_xy(station_catalog['lon'], station_catalog['lat']),
            crs='EPSG:4326',
        )
        seed_thiessen = build_thiessen(seed_points, nyc_union)

        city_station_ids = set(seed_thiessen['station_id'].astype(str))
        city_stations = station_catalog[station_catalog['station_id'].isin(city_station_ids)].copy().sort_values('station_id').reset_index(drop=True)
        if city_stations.empty:
            raise ValueError('No se encontraron estaciones de precipitación dentro de NYC.')

        # Edita esta lista antes de correr la descarga si quieres un subconjunto.
        wanted_station_ids = []
        if wanted_station_ids:
            analysis_stations = city_stations.loc[city_stations['station_id'].isin(wanted_station_ids)].copy()
        else:
            analysis_stations = city_stations.copy()
        analysis_stations = analysis_stations.sort_values('station_id').reset_index(drop=True)
        analysis_points = gpd.GeoDataFrame(
            analysis_stations,
            geometry=gpd.points_from_xy(analysis_stations['lon'], analysis_stations['lat']),
            crs='EPSG:4326',
        )

        raw_precip = load_or_fetch_precip(analysis_stations['station_id'].tolist())
        raw_precip = raw_precip.merge(
            analysis_stations[['station_id', 'station_name', 'lat', 'lon']],
            on='station_id',
            how='left',
        )

        print(f'Estaciones del catalogo: {len(station_catalog):,}')
        print(f'Estaciones que intersectan NYC: {len(city_stations):,}')
        print(f'Estaciones seleccionadas para el analisis: {len(analysis_stations):,}')
        print(f'Registros NOAA descargados: {len(raw_precip):,}')
        raw_precip.head()
        """,
    )

    set_cell(
        nb,
        4,
        """
        # Mapa: red seed, red activa para NYC y estaciones seleccionadas.
        fig, ax = plt.subplots(figsize=(10, 10))
        nyc_boroughs.boundary.plot(ax=ax, color='black', linewidth=1)
        seed_points.plot(ax=ax, color='0.78', markersize=12, alpha=0.45)
        seed_thiessen.plot(ax=ax, color='#fee0d2', alpha=0.35, edgecolor='0.35', linewidth=0.8)
        analysis_points.plot(ax=ax, color='#cb181d', markersize=35)
        set_map_extent(ax, nyc_union, pad=0.04)
        ax.set_title('NOAA precipitation stations and Thiessen cells clipped to NYC', fontsize=13)
        ax.set_axis_off()
        plt.show()

        seed_thiessen[['station_id', 'station_name', 'geometry']]
        """,
    )

    set_cell(
        nb,
        5,
        """
        # Guardar capas espaciales.
        seed_points_out = SPATIAL_OUTPUT_DIR / 'precipitation_seed_stations.geojson'
        city_points_out = SPATIAL_OUTPUT_DIR / 'precipitation_city_stations.geojson'
        analysis_points_out = SPATIAL_OUTPUT_DIR / 'precipitation_selected_stations.geojson'
        thiessen_out = SPATIAL_OUTPUT_DIR / 'precipitation_thiessen.geojson'

        seed_points.to_file(seed_points_out, driver='GeoJSON')
        city_points = gpd.GeoDataFrame(
            city_stations,
            geometry=gpd.points_from_xy(city_stations['lon'], city_stations['lat']),
            crs='EPSG:4326',
        )
        city_points.to_file(city_points_out, driver='GeoJSON')
        analysis_points.to_file(analysis_points_out, driver='GeoJSON')
        seed_thiessen.to_file(thiessen_out, driver='GeoJSON')

        print(seed_points_out)
        print(city_points_out)
        print(analysis_points_out)
        print(thiessen_out)
        """,
    )

    set_cell(
        nb,
        6,
        """
        # Segmentacion de eventos por estacion y exportacion de CSV por evento.
        stations_for_events = analysis_stations.copy()
        if stations_for_events.empty:
            raise ValueError('No hay estaciones seleccionadas para segmentar eventos.')

        event_tables = []
        for row in stations_for_events.to_dict('records'):
            station_id = row['station_id']
            station_name = row['station_name']
            series = raw_precip.loc[raw_precip['station_id'] == station_id].copy()
            if series.empty:
                continue
            event_table = split_rain_events(series, station_id, station_name)
            if not event_table.empty:
                event_tables.append(event_table)

        event_summary = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame(
            columns=[
                'station_id', 'station_name', 'event_id', 'event_uid', 'start', 'end',
                'duration_hours', 'precip_mm', 'mean_intensity_mm_per_hr', 'n_records', 'wet_records', 'csv_path',
            ]
        )
        event_summary = event_summary.sort_values(['station_id', 'start', 'event_id'], kind='stable').reset_index(drop=True)
        event_summary.to_csv(EVENT_SUMMARY_PATH, index=False)
        event_summary
        """,
    )

    set_cell(
        nb,
        7,
        """
        # Resumen rapido.
        print(f'Eventos detectados: {len(event_summary):,}')
        display(
            event_summary.groupby('station_id', dropna=False)
            .agg(
                events=('event_id', 'size'),
                total_mm=('precip_mm', 'sum'),
                mean_intensity_mm_per_hr=('mean_intensity_mm_per_hr', 'mean'),
                max_mm=('precip_mm', 'max'),
                median_duration_hours=('duration_hours', 'median'),
                max_records=('n_records', 'max'),
            )
            .sort_values('events', ascending=False)
        )

        display(
            event_summary[['duration_hours', 'precip_mm', 'mean_intensity_mm_per_hr', 'n_records', 'wet_records']]
            .describe()
        )

        print(f'Eventos con mas de un registro: {(event_summary["n_records"] > 1).sum():,}')
        print(f'Eventos con duracion positiva: {(event_summary["duration_hours"] > 0).sum():,}')
        """,
    )

    set_cell(
        nb,
        8,
        """
        # Distribuciones de duracion e intensidad.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        event_summary['duration_hours'].plot.hist(bins=40, ax=axes[0], color='#3B82F6', alpha=0.85)
        axes[0].set_title('Event duration (hours)')
        axes[0].set_xlabel('duration_hours')

        event_summary['mean_intensity_mm_per_hr'].plot.hist(bins=40, ax=axes[1], color='#D97706', alpha=0.85)
        axes[1].set_title('Mean rainfall intensity (mm/hour)')
        axes[1].set_xlabel('mean_intensity_mm_per_hr')

        plt.tight_layout()
        plt.show()

        display(
            event_summary[['station_id', 'event_id', 'start', 'end', 'duration_hours', 'precip_mm', 'mean_intensity_mm_per_hr', 'n_records']]
            .head(10)
        )
        """,
    )

    set_cell(
        nb,
        9,
        """
        # Estaciones con mas eventos.
        station_rank = (
            event_summary.groupby(['station_id', 'station_name'], dropna=False)
            .agg(
                events=('event_id', 'size'),
                total_mm=('precip_mm', 'sum'),
                mean_duration_hours=('duration_hours', 'mean'),
                mean_intensity_mm_per_hr=('mean_intensity_mm_per_hr', 'mean'),
            )
            .reset_index()
            .sort_values(['events', 'total_mm'], ascending=[False, False])
        )
        display(station_rank)
        """,
    )

    set_cell(
        nb,
        10,
        """
        # Filtro manual de estaciones.
        # Edita wanted_station_ids para quedarte solo con las estaciones que quieres estudiar.
        # Si cambias esta lista, vuelve a ejecutar la celda de descarga NOAA y la segmentacion de eventos.
        wanted_station_ids = [
            # 'GHCND:USW00014732',
            # 'GHCND:USW00094728',
        ]

        if wanted_station_ids:
            analysis_stations = city_stations.loc[city_stations['station_id'].isin(wanted_station_ids)].copy()
        else:
            analysis_stations = city_stations.copy()
        analysis_stations = analysis_stations.sort_values('station_id').reset_index(drop=True)
        analysis_points = gpd.GeoDataFrame(
            analysis_stations,
            geometry=gpd.points_from_xy(analysis_stations['lon'], analysis_stations['lat']),
            crs='EPSG:4326',
        )

        print(f'Estaciones candidatas en NYC: {len(city_stations):,}')
        print(f'Estaciones seleccionadas para el analisis: {len(analysis_stations):,}')
        display(analysis_stations[['station_id', 'station_name', 'lat', 'lon']])
        """,
    )

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def patch_precipitation_notebook() -> None:
    """Patch the current precipitation notebook in-place."""
    nb = json.loads(NOTEBOOK_PATH.read_text())

    def patch_text(text: str) -> str:
        updated = text

        updated = updated.replace(
            "NOAA_TOKEN_ENV_NAMES = ('NOAA_CDO_TOKEN', 'NOAA_API_TOKEN')",
            "NOAA_TOKEN_ENV_NAMES = ('NOAA_CDO_TOKEN', 'NOAA_API_TOKEN', 'NOAA_TOKEN')",
            1,
        )

        updated = updated.replace(
            "HOURLY_DRY_GAP = pd.Timedelta(hours=6)\nLOCAL_CRS = 'EPSG:2263'",
            "HOURLY_DRY_GAP = pd.Timedelta(hours=6)\nFORCE_REFRESH = True\nLOCAL_CRS = 'EPSG:2263'",
            1,
        )

        updated = updated.replace(
            "PROBE_LIMIT = 25\nAPI_MAX_RETRIES = 5",
            "PROBE_LIMIT = 25\nDOWNLOAD_WINDOW_DAYS = 364\nPROBE_WINDOW_DAYS = 90\nAPI_MAX_RETRIES = 5",
            1,
        )

        updated = updated.replace(
            "ROOT = find_project_root()\nINVENTORY_PATH = ROOT / 'data' / 'temporal' / 'noaa' / '4308211.csv'\nNYC_BOUNDARY_PATH = ROOT / 'data' / 'spatial' / 'vector' / 'nyc_borough_boundary' / 'nybb.geojson'\n",
            """ROOT = find_project_root()\n\n\ndef load_env_file(path: Path) -> None:\n    # Load KEY=VALUE pairs from a local .env file.\n    if not path.exists():\n        return\n    for raw_line in path.read_text().splitlines():\n        line = raw_line.strip()\n        if not line or line.startswith('#'):\n            continue\n        if line.startswith('export '):\n            line = line[len('export '):].strip()\n        if '=' not in line:\n            continue\n        key, value = line.split('=', 1)\n        key = key.strip()\n        value = value.strip().strip('\"').strip(\"'\")\n        if key and key not in os.environ:\n            os.environ[key] = value\n\n\nload_env_file(ROOT / '.env')\n\nINVENTORY_PATH = ROOT / 'data' / 'temporal' / 'noaa' / '4308211.csv'\nNYC_BOUNDARY_PATH = ROOT / 'data' / 'spatial' / 'vector' / 'nyc_borough_boundary' / 'nybb.geojson'\n""",
            1,
        )

        updated = updated.replace(
            """def get_noaa_token() -> str | None:\n    for name in NOAA_TOKEN_ENV_NAMES:\n        token = os.getenv(name)\n        if token:\n            return token\n    logger.warning('NOAA token missing. The notebook will use local cache if available; uncached API requests will return empty frames.')\n    return None\n""",
            """def get_noaa_token() -> str:\n    for name in NOAA_TOKEN_ENV_NAMES:\n        token = os.getenv(name)\n        if token:\n            return token\n    raise RuntimeError('Falta el token de NOAA. Define NOAA_CDO_TOKEN, NOAA_API_TOKEN o NOAA_TOKEN en .env o en el entorno.')\n""",
            1,
        )

        updated = updated.replace(
            """def probe_station_source(token: str | None, dataset_id: str, datatype_id: str, anchor_station_id: str, station_name: str, required_resolution: str, notes: str = '') -> dict | None:\n    aliases = station_aliases(anchor_station_id, dataset_id=dataset_id)\n    cache_key = (dataset_id, datatype_id, anchor_station_id, required_resolution, tuple(aliases))\n    if cache_key in PROBE_CACHE:\n        return PROBE_CACHE[cache_key]\n    if not token:\n        PROBE_CACHE[cache_key] = None\n        return None\n\n    errors = []\n    for alias in aliases:\n        try:\n            raw = fetch_cdo_frame(token=token, dataset_id=dataset_id, datatype_id=datatype_id, station_id=alias, start_date=START_DATE, end_date=END_DATE, limit=PROBE_LIMIT, kind='probe')\n        except Exception as exc:\n            errors.append(f'{alias}: {exc}')\n            continue\n        if raw.empty:\n            continue\n        try:\n            prepared = prepare_cdo_frame(raw, dataset_id=dataset_id, datatype_id=datatype_id)\n        except Exception as exc:\n            errors.append(f'{alias}: parse_error={exc}')\n            continue\n        if prepared.empty:\n            continue\n        resolution = infer_temporal_resolution(prepared['timestamp'], dataset_id=dataset_id, datatype_id=datatype_id)\n        if required_resolution == 'daily' and resolution != 'daily':\n            continue\n        if required_resolution == 'hourly' and resolution not in {'hourly', 'subhourly'}:\n            continue\n        result = {\n            'station_id': anchor_station_id,\n            'station_name': station_name,\n            'source_dataset': dataset_id,\n            'data_type': datatype_id,\n            'source_station_id': str(prepared['source_station_id'].dropna().iloc[0]) if prepared['source_station_id'].notna().any() else alias,\n            'source_alias_used': alias,\n            'temporal_resolution': resolution,\n            'record_count_probe': int(raw.attrs.get('record_count', len(prepared))),\n            'sample_start': prepared['timestamp'].min(),\n            'sample_end': prepared['timestamp'].max(),\n            'candidate_notes': notes,\n        }\n        PROBE_CACHE[cache_key] = result\n        return result\n    if errors:\n        logger.info('Probe failed for station=%s dataset=%s datatype=%s examples=%s', anchor_station_id, dataset_id, datatype_id, ' | '.join(errors[:3]))\n    PROBE_CACHE[cache_key] = None\n    return None\n""",
            """def probe_station_source(token: str | None, dataset_id: str, datatype_id: str, anchor_station_id: str, station_name: str, required_resolution: str, notes: str = '', force_refresh: bool = False) -> dict | None:\n    aliases = station_aliases(anchor_station_id, dataset_id=dataset_id)\n    cache_key = (dataset_id, datatype_id, anchor_station_id, required_resolution, tuple(aliases))\n    if not force_refresh and cache_key in PROBE_CACHE:\n        return PROBE_CACHE[cache_key]\n    if not token:\n        PROBE_CACHE[cache_key] = None\n        return None\n\n    errors = []\n    for alias in aliases:\n        try:\n            raw = fetch_cdo_frame(token=token, dataset_id=dataset_id, datatype_id=datatype_id, station_id=alias, start_date=START_DATE, end_date=END_DATE, limit=PROBE_LIMIT, kind='probe', force_refresh=force_refresh)\n        except Exception as exc:\n            errors.append(f'{alias}: {exc}')\n            continue\n        if raw.empty:\n            continue\n        try:\n            prepared = prepare_cdo_frame(raw, dataset_id=dataset_id, datatype_id=datatype_id)\n        except Exception as exc:\n            errors.append(f'{alias}: parse_error={exc}')\n            continue\n        if prepared.empty:\n            continue\n        resolution = infer_temporal_resolution(prepared['timestamp'], dataset_id=dataset_id, datatype_id=datatype_id)\n        if required_resolution == 'daily' and resolution != 'daily':\n            continue\n        if required_resolution == 'hourly' and resolution not in {'hourly', 'subhourly'}:\n            continue\n        result = {\n            'station_id': anchor_station_id,\n            'station_name': station_name,\n            'source_dataset': dataset_id,\n            'data_type': datatype_id,\n            'source_station_id': str(prepared['source_station_id'].dropna().iloc[0]) if prepared['source_station_id'].notna().any() else alias,\n            'source_alias_used': alias,\n            'temporal_resolution': resolution,\n            'record_count_probe': int(raw.attrs.get('record_count', len(prepared))),\n            'sample_start': prepared['timestamp'].min(),\n            'sample_end': prepared['timestamp'].max(),\n            'candidate_notes': notes,\n        }\n        PROBE_CACHE[cache_key] = result\n        return result\n    if errors:\n        logger.info('Probe failed for station=%s dataset=%s datatype=%s examples=%s', anchor_station_id, dataset_id, datatype_id, ' | '.join(errors[:3]))\n    PROBE_CACHE[cache_key] = None\n    return None\n""",
            1,
        )

        updated = updated.replace(
            """def select_precip_source(token: str | None, station_row: pd.Series, kind: str) -> tuple[dict | None, list[str]]:\n    station_id = str(station_row['station_id'])\n    station_name = str(station_row['station_name'])\n    checked = []\n    if kind == 'daily':\n        candidates = DAILY_CANDIDATES\n        required_resolution = 'daily'\n    elif kind == 'hourly':\n        candidates = hourly_candidate_table(token)\n        required_resolution = 'hourly'\n    else:\n        raise ValueError(f'Unknown kind: {kind}')\n\n    for candidate in candidates:\n        dataset_id = candidate['dataset_id']\n        datatype_id = candidate['datatype_id']\n        checked.append(f'{dataset_id}:{datatype_id}')\n        probe = probe_station_source(\n            token=token,\n            dataset_id=dataset_id,\n            datatype_id=datatype_id,\n            anchor_station_id=station_id,\n            station_name=station_name,\n            required_resolution=required_resolution,\n            notes=candidate.get('notes', ''),\n        )\n        if probe is not None:\n            probe['source_kind'] = kind\n            probe['checked_candidates'] = checked.copy()\n            return probe, checked\n    return None, checked\n""",
            """def select_precip_source(token: str | None, station_row: pd.Series, kind: str, force_refresh: bool = False) -> tuple[dict | None, list[str]]:\n    station_id = str(station_row['station_id'])\n    station_name = str(station_row['station_name'])\n    checked = []\n    if kind == 'daily':\n        candidates = DAILY_CANDIDATES\n        required_resolution = 'daily'\n    elif kind == 'hourly':\n        candidates = hourly_candidate_table(token)\n        required_resolution = 'hourly'\n    else:\n        raise ValueError(f'Unknown kind: {kind}')\n\n    for candidate in candidates:\n        dataset_id = candidate['dataset_id']\n        datatype_id = candidate['datatype_id']\n        checked.append(f'{dataset_id}:{datatype_id}')\n        probe = probe_station_source(\n            token=token,\n            dataset_id=dataset_id,\n            datatype_id=datatype_id,\n            anchor_station_id=station_id,\n            station_name=station_name,\n            required_resolution=required_resolution,\n            notes=candidate.get('notes', ''),\n            force_refresh=force_refresh,\n        )\n        if probe is not None:\n            probe['source_kind'] = kind\n            probe['checked_candidates'] = checked.copy()\n            return probe, checked\n    return None, checked\n""",
            1,
        )

        updated = updated.replace(
            """def audit_station_availability(station_row: pd.Series, token: str | None) -> dict:\n    daily, daily_checked = select_precip_source(token, station_row, 'daily')\n    hourly, hourly_checked = select_precip_source(token, station_row, 'hourly')\n""",
            """def audit_station_availability(station_row: pd.Series, token: str | None, force_refresh: bool = False) -> dict:\n    daily, daily_checked = select_precip_source(token, station_row, 'daily', force_refresh=force_refresh)\n    hourly, hourly_checked = select_precip_source(token, station_row, 'hourly', force_refresh=force_refresh)\n""",
            1,
        )

        updated = updated.replace(
            """token = get_noaa_token()\nnyc_boroughs, nyc_union = load_nyc_boundary()\nstation_inventory = load_station_inventory(TARGET_STATIONS)\n""",
            """token = get_noaa_token()\nnyc_boroughs, nyc_union = load_nyc_boundary()\nPRECIP_DATATYPE_CACHE.clear()\nPROBE_CACHE.clear()\nstation_inventory = load_station_inventory(TARGET_STATIONS)\n""",
            1,
        )

        updated = updated.replace(
            "availability_probe = pd.DataFrame([audit_station_availability(row, token) for row in station_inventory.to_dict('records')])",
            "availability_probe = pd.DataFrame([audit_station_availability(row, token, force_refresh=FORCE_REFRESH) for row in station_inventory.to_dict('records')])",
            1,
        )

        updated = updated.replace(
            """def fetch_selected_series(selection: pd.DataFrame, kind: str, token: str | None) -> pd.DataFrame:\n    frames = []\n    for row in selection.to_dict('records'):\n        source_dataset = row[f'{kind}_source_dataset']\n        data_type = row[f'{kind}_data_type']\n        source_station_id = row[f'{kind}_source_station_id']\n        if pd.isna(source_dataset) or pd.isna(data_type) or pd.isna(source_station_id):\n            continue\n        try:\n            raw = fetch_cdo_frame(token=token, dataset_id=str(source_dataset), datatype_id=str(data_type), station_id=str(source_station_id), start_date=START_DATE, end_date=END_DATE, limit=API_LIMIT, kind=kind)\n        except Exception as exc:\n            logger.warning('%s: failed to fetch station=%s dataset=%s datatype=%s: %s', kind, row['station_id'], source_dataset, data_type, exc)\n            continue\n        if raw.empty:\n            logger.warning('%s: no data fetched for station=%s dataset=%s datatype=%s', kind, row['station_id'], source_dataset, data_type)\n            continue\n        try:\n            frame = finalize_record_frame(raw, anchor_station_id=row['station_id'], station_name=row['station_name'], source_dataset=str(source_dataset), data_type=str(data_type), source_station_id=str(source_station_id), analysis_kind=kind)\n        except Exception as exc:\n            logger.warning('%s: fetched data failed validation for station=%s dataset=%s datatype=%s: %s', kind, row['station_id'], source_dataset, data_type, exc)\n            continue\n        frames.append(frame)\n    if not frames:\n        return pd.DataFrame(columns=['station_id', 'station_name', 'source_station_id', 'source_dataset', 'data_type', 'analysis_kind', 'temporal_resolution', 'timestamp', 'precip_mm', 'year', 'month', 'day'])\n    return validate_series(pd.concat(frames, ignore_index=True), f'{kind}_records')\n""",
            """def fetch_selected_series(selection: pd.DataFrame, kind: str, token: str | None, force_refresh: bool = False) -> pd.DataFrame:\n    frames = []\n    for row in selection.to_dict('records'):\n        source_dataset = row[f'{kind}_source_dataset']\n        data_type = row[f'{kind}_data_type']\n        source_station_id = row[f'{kind}_source_station_id']\n        if pd.isna(source_dataset) or pd.isna(data_type) or pd.isna(source_station_id):\n            continue\n        try:\n            raw = fetch_cdo_frame(token=token, dataset_id=str(source_dataset), datatype_id=str(data_type), station_id=str(source_station_id), start_date=START_DATE, end_date=END_DATE, limit=API_LIMIT, kind=kind, force_refresh=force_refresh)\n        except Exception as exc:\n            logger.warning('%s: failed to fetch station=%s dataset=%s datatype=%s: %s', kind, row['station_id'], source_dataset, data_type, exc)\n            continue\n        if raw.empty:\n            logger.warning('%s: no data fetched for station=%s dataset=%s datatype=%s', kind, row['station_id'], source_dataset, data_type)\n            continue\n        try:\n            frame = finalize_record_frame(raw, anchor_station_id=row['station_id'], station_name=row['station_name'], source_dataset=str(source_dataset), data_type=str(data_type), source_station_id=str(source_station_id), analysis_kind=kind)\n        except Exception as exc:\n            logger.warning('%s: fetched data failed validation for station=%s dataset=%s datatype=%s: %s', kind, row['station_id'], source_dataset, data_type, exc)\n            continue\n        frames.append(frame)\n    if not frames:\n        return pd.DataFrame(columns=['station_id', 'station_name', 'source_station_id', 'source_dataset', 'data_type', 'analysis_kind', 'temporal_resolution', 'timestamp', 'precip_mm', 'year', 'month', 'day'])\n    return validate_series(pd.concat(frames, ignore_index=True), f'{kind}_records')\n""",
            1,
        )

        updated = updated.replace(
            "daily_records = fetch_selected_series(daily_selection, 'daily', token) if not daily_selection.empty else pd.DataFrame(columns=['station_id', 'station_name', 'source_station_id', 'source_dataset', 'data_type', 'analysis_kind', 'temporal_resolution', 'timestamp', 'precip_mm', 'year', 'month', 'day'])",
            "daily_records = fetch_selected_series(daily_selection, 'daily', token, force_refresh=FORCE_REFRESH) if not daily_selection.empty else pd.DataFrame(columns=['station_id', 'station_name', 'source_station_id', 'source_dataset', 'data_type', 'analysis_kind', 'temporal_resolution', 'timestamp', 'precip_mm', 'year', 'month', 'day'])",
            1,
        )

        updated = updated.replace(
            "hourly_records = fetch_selected_series(hourly_selection, 'hourly', token) if not hourly_selection.empty else pd.DataFrame(columns=['station_id', 'station_name', 'source_station_id', 'source_dataset', 'data_type', 'analysis_kind', 'temporal_resolution', 'timestamp', 'precip_mm', 'year', 'month', 'day'])",
            "hourly_records = fetch_selected_series(hourly_selection, 'hourly', token, force_refresh=FORCE_REFRESH) if not hourly_selection.empty else pd.DataFrame(columns=['station_id', 'station_name', 'source_station_id', 'source_dataset', 'data_type', 'analysis_kind', 'temporal_resolution', 'timestamp', 'precip_mm', 'year', 'month', 'day'])",
            1,
        )

        updated = updated.replace(
            """# Edita esta lista antes de correr la descarga si quieres un subconjunto.\n        wanted_station_ids = []\n""",
            """# Edita esta lista antes de correr la descarga si quieres un subconjunto.\n        # Si cambias el subconjunto, vuelve a ejecutar desde esta celda para forzar refresh.\n        wanted_station_ids = []\n""",
            1,
        )

        if updated != text:
            cell["source"] = [f"{line}\n" for line in updated.rstrip("\n").splitlines()]
            cell["outputs"] = []
            cell["execution_count"] = None

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    patch_precipitation_notebook()
