from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/census.ipynb")


def md_cell(text: str) -> dict:
    text = dedent(text).strip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict:
    text = dedent(text).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    nb["cells"] = [
        md_cell(
            """
            # URBAN FLOOD STRESS | Census

            Construye capas censales ACS 5-Year por periodo para NYC, unidas a geometrías TIGER/Line y listas para análisis espacial posterior.

            Flujo:
            - descargar ACS 5-Year para cada periodo,
            - descargar TIGER/Line de tracts para el mismo año TIGER,
            - unir atributos y geometría por `GEOID`,
            - etiquetar boroughs de NYC,
            - guardar un GeoPackage por periodo y un CSV de metadata.
            """
        ),
        md_cell(
            """
            ## Variables ACS incluidas

            El notebook recupera tres familias de productos ACS para cubrir vulnerabilidad social, capacidad de recuperación, evacuación e intervención:

            - `detailed`:
              - `B17001`, `B19013`, `B19057`, `B22001`, `B15001`, `B05002`, `B25044`, `B25003`, `B25014`, `B25024`, `B25034`, `B28002`, `B11007`, `C16002`
            - `subject`:
              - `S1701`, `S1501`, `S1810`
            - `profile`:
              - `DP02`, `DP03`, `DP04`

            El enfoque usa tablas completas por grupo, luego conserva columnas de estimación y porcentaje para mantener el dataset útil sin cargar anotaciones innecesarias.

            ## Por qué ACS 5-Year

            ACS 5-Year da estimaciones más estables a nivel tracto. Para este proyecto importa más la comparabilidad y cobertura espacial que la variación anual de una sola muestra.

            ## Por qué TIGER/Line debe coincidir

            `GEOID` funciona como llave, pero el significado geométrico del tracto cambia con el vintage cartográfico. Por eso:

            - `2020–2024`, `2019–2023`, `2018–2022`, `2017–2021`, `2016–2020` usan `Census 2020 geography`,
            - `2015–2019` y periodos anteriores usan `Census 2010 geography`.

            El notebook empareja cada ACS period con su TIGER year correspondiente.

            ## Relevancia socioespacial

            Estas variables ayudan a modelar:

            - pobreza y capacidad de recuperación,
            - educación e intermediación institucional,
            - vehículos, tenencia y hacinamiento para evacuación,
            - conectividad, discapacidad y hogares de adultos mayores para respuesta de emergencia.
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import logging
            import os
            import sys
            import zipfile
            from functools import reduce
            from pathlib import Path
            from typing import Iterable

            import geopandas as gpd
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import requests
            from shapely import make_valid
            from shapely.ops import unary_union

            try:
                from IPython.display import display
            except ImportError:  # pragma: no cover - notebook fallback
                display = print

            MPLCONFIGDIR = Path("/tmp/matplotlib")
            MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
            plt.style.use("seaborn-v0_8-whitegrid")

            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
            logger = logging.getLogger("census_notebook")

            def find_project_root(start: Path | None = None) -> Path:
                current = start or Path.cwd()
                for candidate in [current, *current.parents]:
                    if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
                        return candidate
                raise FileNotFoundError("No se encontro la raiz del proyecto.")


            ROOT = find_project_root()
            SRC = ROOT / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))

            from project_name.utils import ensure_directory

            ACS_BASE_URL = "https://api.census.gov/data"
            CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
            NY_STATE_FIPS = "36"
            NYC_COUNTIES = {
                "Bronx": "005",
                "Brooklyn": "047",
                "Manhattan": "061",
                "Queens": "081",
                "Staten Island": "085",
            }

            FORCE_REFRESH = False
            RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
            REQUEST_TIMEOUT = 120
            MAX_RETRIES = 5
            RETRY_BACKOFF_SECONDS = 2.0

            NYC_BOUNDARY_PATH = ROOT / "data" / "spatial" / "vector" / "nyc_borough_boundary" / "nybb.geojson"
            RAW_CENSUS_DIR = ensure_directory(ROOT / "data" / "raw" / "census")
            PROCESSED_CENSUS_DIR = ensure_directory(ROOT / "data" / "processed" / "census")
            TIGER_CACHE_DIR = ensure_directory(RAW_CENSUS_DIR / "tiger")
            ACS_CACHE_DIR = ensure_directory(RAW_CENSUS_DIR / "acs")
            PROCESSING_LOG_PATH = PROCESSED_CENSUS_DIR / "census_processing_log.csv"
            METADATA_PATH = PROCESSED_CENSUS_DIR / "census_period_metadata.csv"

            ACS_PERIOD_ROWS = [
                ("2006_2010", 2010, 2010, "census_2010_geography"),
                ("2007_2011", 2011, 2011, "census_2010_geography"),
                ("2008_2012", 2012, 2012, "census_2010_geography"),
                ("2009_2013", 2013, 2013, "census_2010_geography"),
                ("2010_2014", 2014, 2014, "census_2010_geography"),
                ("2011_2015", 2015, 2015, "census_2010_geography"),
                ("2012_2016", 2016, 2016, "census_2010_geography"),
                ("2013_2017", 2017, 2017, "census_2010_geography"),
                ("2014_2018", 2018, 2018, "census_2010_geography"),
                ("2015_2019", 2019, 2019, "census_2010_geography"),
                ("2016_2020", 2020, 2020, "census_2020_geography"),
                ("2017_2021", 2021, 2021, "census_2020_geography"),
                ("2018_2022", 2022, 2022, "census_2020_geography"),
                ("2019_2023", 2023, 2023, "census_2020_geography"),
                ("2020_2024", 2024, 2024, "census_2020_geography"),
            ]
            ACS_PERIOD_CONFIGS = {
                period: {
                    "acs_year": acs_year,
                    "tiger_year": tiger_year,
                    "census_geography_base": census_geography_base,
                }
                for period, acs_year, tiger_year, census_geography_base in ACS_PERIOD_ROWS
            }

            ACS_DATASET_SPECS = {
                "detailed": {
                    "dataset_path": "acs/acs5",
                    "tables": [
                        "B17001",
                        "B19013",
                        "B19057",
                        "B22001",
                        "B15001",
                        "B05002",
                        "B25044",
                        "B25003",
                        "B25014",
                        "B25024",
                        "B25034",
                        "B28002",
                        "B11007",
                        "C16002",
                    ],
                },
                "subject": {
                    "dataset_path": "acs/acs5/subject",
                    "tables": ["S1701", "S1501", "S1810"],
                },
                "profile": {
                    "dataset_path": "acs/acs5/profile",
                    "tables": ["DP02", "DP03", "DP04"],
                },
            }

            pd.set_option("display.max_columns", 200)
            pd.set_option("display.width", 160)
            print(ROOT)
            print(PROCESSED_CENSUS_DIR)
            print(RAW_CENSUS_DIR)
            print(ACS_PERIOD_CONFIGS["2020_2024"])
            """
        ),
        code_cell(
            """
            def first_existing_column(columns: Iterable[str], *candidates: str) -> str | None:
                column_set = {str(column) for column in columns}
                for candidate in candidates:
                    if candidate in column_set:
                        return candidate
                return None


            def safe_ratio(
                numerator: pd.Series | None,
                denominator: pd.Series | None,
                index: pd.Index | None = None,
            ) -> pd.Series:
                if numerator is None or denominator is None:
                    if index is None:
                        return pd.Series(dtype="float64")
                    return pd.Series(np.nan, index=index, dtype="float64")
                numerator = pd.to_numeric(numerator, errors="coerce")
                denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
                return numerator / denominator


            def path_for_table_cache(acs_period: str, dataset_kind: str, table_code: str) -> Path:
                cache_dir = ACS_CACHE_DIR / acs_period / dataset_kind
                cache_dir.mkdir(parents=True, exist_ok=True)
                return cache_dir / f"{table_code}.csv"


            def path_for_tiger_cache(tiger_year: int) -> Path:
                TIGER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                return TIGER_CACHE_DIR / f"tl_{tiger_year}_{NY_STATE_FIPS}_tract.zip"


            def request_json(session: requests.Session, url: str, params: dict[str, str], *, max_retries: int = MAX_RETRIES) -> list:
                last_error: Exception | None = None
                for attempt in range(max_retries + 1):
                    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                    if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                        wait_seconds = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                        logger.warning(
                            "Retrying request %s status=%s attempt=%s/%s sleep=%.1fs",
                            url,
                            response.status_code,
                            attempt + 1,
                            max_retries + 1,
                            wait_seconds,
                        )
                        import time

                        time.sleep(wait_seconds)
                        continue

                    try:
                        response.raise_for_status()
                    except Exception as exc:
                        body = response.text.strip().replace("\\n", " ")[:500]
                        last_error = RuntimeError(
                            f"Request failed for {url}: {response.status_code}. {body}"
                        )
                        raise last_error from exc

                    try:
                        payload = response.json()
                    except Exception as exc:
                        body = response.text.strip().replace("\\n", " ")[:500]
                        last_error = RuntimeError(f"Non-JSON response for {url}: {body}")
                        raise last_error from exc

                    if isinstance(payload, dict) and "error" in payload:
                        raise RuntimeError(f"API error for {url}: {payload['error']}")
                    return payload

                if last_error is not None:
                    raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error
                raise RuntimeError(f"Unable to fetch {url}")


            def fetch_acs_group(
                acs_year: int,
                dataset_path: str,
                table_code: str,
                county_map: dict[str, str],
                acs_period: str,
                api_key: str | None = None,
                force_refresh: bool = FORCE_REFRESH,
            ) -> pd.DataFrame | None:
                cache_path = path_for_table_cache(acs_period, dataset_path.replace("/", "_"), table_code)
                if cache_path.exists() and not force_refresh:
                    frame = pd.read_csv(cache_path, low_memory=False)
                    logger.info("Loaded cached ACS table %s for %s", table_code, acs_period)
                    return frame

                url = f"{ACS_BASE_URL}/{acs_year}/{dataset_path}"
                session = requests.Session()
                frames = []

                for county_name, county_fips in county_map.items():
                    params = {
                        "get": f"NAME,group({table_code})",
                        "for": "tract:*",
                        "in": f"state:{NY_STATE_FIPS} county:{county_fips}",
                    }
                    if api_key:
                        params["key"] = api_key

                    try:
                        payload = request_json(session, url, params)
                        header, rows = payload[0], payload[1:]
                        county_frame = pd.DataFrame(rows, columns=header)
                        county_frame["county_name"] = county_name
                        county_frame["acs_table"] = table_code
                        county_frame["dataset_path"] = dataset_path
                        county_frame["acs_year"] = acs_year
                        county_frame["acs_period"] = acs_period
                        frames.append(county_frame)
                    except Exception as exc:
                        logger.warning(
                            "Skipping ACS table %s for %s county=%s: %s",
                            table_code,
                            acs_period,
                            county_name,
                            exc,
                        )

                if not frames:
                    return None

                combined = pd.concat(frames, ignore_index=True)
                combined.to_csv(cache_path, index=False)
                return combined


            def download_tiger_tracts(tiger_year: int, force_refresh: bool = FORCE_REFRESH) -> Path:
                zip_path = path_for_tiger_cache(tiger_year)
                if zip_path.exists() and not force_refresh:
                    return zip_path

                url = (
                    f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/TRACT/"
                    f"tl_{tiger_year}_{NY_STATE_FIPS}_tract.zip"
                )
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                zip_path.write_bytes(response.content)
                return zip_path


            def read_tiger_zip(zip_path: Path) -> gpd.GeoDataFrame:
                with zipfile.ZipFile(zip_path) as zip_file:
                    shp_name = next(name for name in zip_file.namelist() if name.endswith(".shp"))
                virtual_path = f"zip://{zip_path}!{shp_name}"
                tracts = gpd.read_file(virtual_path)
                tracts = tracts.copy()
                tracts["geometry"] = tracts.geometry.apply(make_valid)
                tracts = tracts.to_crs(epsg=2263)

                geoid_col = first_existing_column(tracts.columns, "GEOID", "GEOIDFQ", "GEOID10")
                if geoid_col is None:
                    raise KeyError("No se encontro GEOID en el TIGER shapefile")
                if geoid_col != "GEOID":
                    tracts = tracts.rename(columns={geoid_col: "GEOID"})
                tracts["GEOID"] = tracts["GEOID"].astype(str)
                return tracts


            def normalize_acs_frame(frame: pd.DataFrame) -> pd.DataFrame:
                normalized = frame.copy()
                normalized["state"] = normalized["state"].astype(str).str.zfill(2)
                normalized["county"] = normalized["county"].astype(str).str.zfill(3)
                normalized["tract"] = normalized["tract"].astype(str).str.zfill(6)
                normalized["GEOID"] = normalized["state"] + normalized["county"] + normalized["tract"]

                keep_columns = ["GEOID"] + [
                    column
                    for column in normalized.columns
                    if column not in {"NAME", "state", "county", "tract", "GEOID", "county_name"}
                    and (column.endswith("E") or column.endswith("PE"))
                    and not column.endswith(("EA", "MA"))
                ]
                normalized = normalized[keep_columns].copy()

                for column in keep_columns:
                    if column != "GEOID":
                        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
                return normalized


            def merge_acs_tables(table_frames: list[pd.DataFrame]) -> pd.DataFrame:
                if not table_frames:
                    return pd.DataFrame(columns=["GEOID"])

                cleaned_frames = []
                for frame in table_frames:
                    frame = frame.copy()
                    frame = frame.loc[:, ~frame.columns.duplicated()]
                    if "GEOID" not in frame.columns:
                        continue
                    cleaned_frames.append(frame)

                if not cleaned_frames:
                    return pd.DataFrame(columns=["GEOID"])

                merged = reduce(lambda left, right: left.merge(right, on="GEOID", how="outer"), cleaned_frames)
                merged["GEOID"] = merged["GEOID"].astype(str)
                return merged


            def add_derived_metrics(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
                enriched = frame.copy()
                enriched["poverty_rate"] = safe_ratio(
                    enriched.get("B17001_002E"),
                    enriched.get("B17001_001E"),
                    enriched.index,
                )
                enriched["renter_share"] = safe_ratio(
                    enriched.get("B25003_003E"),
                    enriched.get("B25003_001E"),
                    enriched.index,
                )
                enriched["no_vehicle_share"] = safe_ratio(
                    enriched.get("B25044_003E"),
                    enriched.get("B25044_001E"),
                    enriched.index,
                )
                return enriched


            def load_nyc_boundary() -> gpd.GeoDataFrame:
                boundary = gpd.read_file(NYC_BOUNDARY_PATH)
                boundary = boundary.copy()
                boundary["geometry"] = boundary.geometry.apply(make_valid)
                boundary = boundary.to_crs(epsg=2263)
                return boundary


            def borough_column(boundary: gpd.GeoDataFrame) -> str:
                column = first_existing_column(boundary.columns, "BoroName", "borough", "boro_name", "borough_name")
                if column is None:
                    raise KeyError("No borough name column found in nybb.geojson")
                return column


            def attach_borough_context(tracts: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
                borough_field = borough_column(boundary)
                boroughs = boundary[[borough_field, "geometry"]].rename(columns={borough_field: "borough"})
                joined = gpd.sjoin(tracts, boroughs, how="left", predicate="intersects")
                if "index_right" in joined.columns:
                    joined = joined.drop(columns=["index_right"])
                joined = joined.drop_duplicates(subset=["GEOID"]).copy()
                return joined


            def finalize_layer(
                combined_acs: pd.DataFrame,
                tiger_tracts: gpd.GeoDataFrame,
                boundary: gpd.GeoDataFrame,
                acs_period: str,
                acs_year: int,
                tiger_year: int,
                census_geography_base: str,
            ) -> gpd.GeoDataFrame:
                merged = tiger_tracts.merge(combined_acs, on="GEOID", how="inner")
                merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=tiger_tracts.crs)
                merged = attach_borough_context(merged, boundary)
                merged = add_derived_metrics(merged)
                merged["acs_period"] = acs_period
                merged["acs_year"] = acs_year
                merged["tiger_year"] = tiger_year
                merged["census_geography_base"] = census_geography_base
                merged["census_processed"] = True
                return merged


            def summarize_layer(layer: gpd.GeoDataFrame, output_path: Path, acs_period: str, acs_year: int, tiger_year: int, census_geography_base: str) -> dict:
                metadata_columns = {
                    "GEOID",
                    "borough",
                    "acs_period",
                    "acs_year",
                    "tiger_year",
                    "census_geography_base",
                    "census_processed",
                    "geometry",
                }
                variable_columns = [column for column in layer.columns if column not in metadata_columns]
                return {
                    "acs_period": acs_period,
                    "acs_year": acs_year,
                    "tiger_year": tiger_year,
                    "census_geography_base": census_geography_base,
                    "number_of_features": int(len(layer)),
                    "number_of_variables": int(len(variable_columns)),
                    "output_path": str(output_path),
                }


            def build_period_layer(
                acs_period: str,
                period_config: dict[str, object],
                force_refresh: bool = FORCE_REFRESH,
            ) -> tuple[gpd.GeoDataFrame, dict, list[dict]]:
                acs_year = int(period_config["acs_year"])
                tiger_year = int(period_config["tiger_year"])
                census_geography_base = str(period_config["census_geography_base"])
                output_path = PROCESSED_CENSUS_DIR / f"acs_{acs_period}_tiger{tiger_year}.gpkg"
                layer_log: list[dict] = []

                if output_path.exists() and not force_refresh:
                    layer = gpd.read_file(output_path)
                    summary = summarize_layer(layer, output_path, acs_period, acs_year, tiger_year, census_geography_base)
                    logger.info("Loaded cached period %s", acs_period)
                    return layer, summary, layer_log

                table_frames: list[pd.DataFrame] = []
                for dataset_kind, spec in ACS_DATASET_SPECS.items():
                    dataset_path = str(spec["dataset_path"])
                    for table_code in spec["tables"]:
                        raw_frame = fetch_acs_group(
                            acs_year=acs_year,
                            dataset_path=dataset_path,
                            table_code=table_code,
                            county_map=NYC_COUNTIES,
                            acs_period=acs_period,
                            api_key=CENSUS_API_KEY,
                            force_refresh=force_refresh,
                        )
                        if raw_frame is None or raw_frame.empty:
                            layer_log.append(
                                {
                                    "acs_period": acs_period,
                                    "acs_year": acs_year,
                                    "tiger_year": tiger_year,
                                    "dataset_kind": dataset_kind,
                                    "table_code": table_code,
                                    "status": "missing",
                                    "message": "Table unavailable or empty",
                                }
                            )
                            continue

                        normalized = normalize_acs_frame(raw_frame)
                        normalized["acs_period"] = acs_period
                        normalized["acs_year"] = acs_year
                        normalized["tiger_year"] = tiger_year
                        normalized["census_geography_base"] = census_geography_base
                        table_frames.append(normalized)
                        layer_log.append(
                            {
                                "acs_period": acs_period,
                                "acs_year": acs_year,
                                "tiger_year": tiger_year,
                                "dataset_kind": dataset_kind,
                                "table_code": table_code,
                                "status": "downloaded",
                                "message": "OK",
                            }
                        )

                combined = merge_acs_tables(table_frames)
                if combined.empty:
                    raise RuntimeError(f"No se pudieron descargar tablas ACS para {acs_period}.")

                tiger_zip = download_tiger_tracts(tiger_year, force_refresh=force_refresh)
                tiger_tracts = read_tiger_zip(tiger_zip)
                boundary = load_nyc_boundary()
                layer = finalize_layer(
                    combined_acs=combined,
                    tiger_tracts=tiger_tracts,
                    boundary=boundary,
                    acs_period=acs_period,
                    acs_year=acs_year,
                    tiger_year=tiger_year,
                    census_geography_base=census_geography_base,
                )

                layer.to_file(output_path, driver="GPKG")
                summary = summarize_layer(layer, output_path, acs_period, acs_year, tiger_year, census_geography_base)
                return layer, summary, layer_log
            """
        ),
        md_cell(
            """
            ## Procesamiento

            Cada periodo se procesa de forma independiente. Si una tabla no existe en ese año o el API devuelve error, el notebook lo registra y continúa con el resto.

            Los outputs quedan en:
            - `data/processed/census/acs_<period>_tiger<year>.gpkg`
            - `data/processed/census/census_period_metadata.csv`
            - `data/processed/census/census_processing_log.csv`
            """
        ),
        code_cell(
            """
            period_results = []
            issue_records = []

            for acs_period, period_config in ACS_PERIOD_CONFIGS.items():
                logger.info(
                    "Processing period %s (ACS %s / TIGER %s)",
                    acs_period,
                    period_config["acs_year"],
                    period_config["tiger_year"],
                )
                try:
                    layer, summary, layer_log = build_period_layer(acs_period, period_config)
                    period_results.append(
                        {
                            "acs_period": acs_period,
                            "layer": layer,
                            "summary": summary,
                        }
                    )
                    issue_records.extend(layer_log)
                except Exception as exc:
                    logger.exception("Failed to process %s: %s", acs_period, exc)
                    issue_records.append(
                        {
                            "acs_period": acs_period,
                            "acs_year": period_config["acs_year"],
                            "tiger_year": period_config["tiger_year"],
                            "dataset_kind": "all",
                            "table_code": None,
                            "status": "failed",
                            "message": str(exc),
                        }
                    )

            metadata_columns = [
                "acs_period",
                "acs_year",
                "tiger_year",
                "census_geography_base",
                "number_of_features",
                "number_of_variables",
                "output_path",
            ]
            if period_results:
                metadata_frame = pd.DataFrame([item["summary"] for item in period_results]).sort_values(
                    ["acs_year", "tiger_year", "acs_period"]
                )
            else:
                metadata_frame = pd.DataFrame(columns=metadata_columns)
            metadata_frame.to_csv(METADATA_PATH, index=False)

            issues_frame = pd.DataFrame(issue_records)
            if not issues_frame.empty:
                issues_frame.to_csv(PROCESSING_LOG_PATH, index=False)
            else:
                pd.DataFrame(
                    columns=[
                        "acs_period",
                        "acs_year",
                        "tiger_year",
                        "dataset_kind",
                        "table_code",
                        "status",
                        "message",
                    ]
                ).to_csv(PROCESSING_LOG_PATH, index=False)

            print(f"periods processed: {len(period_results):,}")
            print(f"metadata saved: {METADATA_PATH}")
            print(f"processing log saved: {PROCESSING_LOG_PATH}")
            display(metadata_frame)
            if not issues_frame.empty:
                display(issues_frame)
            """
        ),
        code_cell(
            """
            if period_results:
                latest_result = period_results[-1]
                latest_layer = latest_result["layer"]
                latest_period = latest_result["acs_period"]

                fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)

                latest_layer.plot(
                    ax=axes[0],
                    column="median_household_income",
                    cmap="viridis",
                    linewidth=0.05,
                    edgecolor="white",
                    legend=True,
                    legend_kwds={"label": "Median household income"},
                )
                boundary = load_nyc_boundary()
                boundary.boundary.plot(ax=axes[0], color="black", linewidth=0.7)
                axes[0].set_title(f"Median household income by tract ({latest_period})")
                axes[0].set_axis_off()

                latest_layer.plot(
                    ax=axes[1],
                    column="poverty_rate",
                    cmap="magma_r",
                    linewidth=0.05,
                    edgecolor="white",
                    legend=True,
                    legend_kwds={"label": "Poverty rate"},
                )
                boundary.boundary.plot(ax=axes[1], color="black", linewidth=0.7)
                axes[1].set_title(f"Poverty rate by tract ({latest_period})")
                axes[1].set_axis_off()

                plt.show()

                fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
                latest_layer["median_household_income"].dropna().plot.hist(ax=axes[0], bins=30, color="#2c7fb8", alpha=0.85)
                axes[0].set_title("Median household income distribution")
                axes[0].set_xlabel("USD")
                axes[0].set_ylabel("Count")

                latest_layer["poverty_rate"].dropna().plot.hist(ax=axes[1], bins=30, color="#d95f0e", alpha=0.85)
                axes[1].set_title("Poverty rate distribution")
                axes[1].set_xlabel("Share")
                axes[1].set_ylabel("Count")

                plt.show()

                display(
                    latest_layer[
                        [
                            "GEOID",
                            "borough",
                            "acs_period",
                            "median_household_income",
                            "poverty_rate",
                            "renter_share",
                            "no_vehicle_share",
                            "geometry",
                        ]
                    ].head()
                )
            else:
                print("No period layers were generated.")
            """
        ),
        md_cell(
            """
            ## Uso

            - Define `CENSUS_API_KEY` si quieres evitar límites de tasa de la API.
            - Cambia `FORCE_REFRESH = True` si necesitas reconstruir todo desde cero.
            - Si una tabla no existe en cierto año, el notebook la reporta y sigue.

            El resultado final queda listo para cruces espaciales posteriores con inundación, red vial y 311.
            """
        ),
    ]
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
