from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/infrastructure.ipynb")


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


def build_notebook() -> dict:
    return {
        "cells": [
            md_cell(
                """
                # URBAN FLOOD STRESS | Infrastructure

                Descarga y resume capas de infraestructura urbana relevantes para drenaje y respuesta hidrometeorológica en NYC.

                Capas incluidas:
                - catch basins
                - green infrastructure
                - outfalls

                El notebook guarda los CSV crudos y los GeoPackages procesados en:
                - `data/raw/infrastructure/`
                - `data/processed/infrastructure/`
                """
            ),
            code_cell(
                """
                import os
                import sys
                from pathlib import Path

                import geopandas as gpd
                import matplotlib.pyplot as plt
                import pandas as pd
                from IPython.display import display

                ROOT = Path.cwd().resolve()
                while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
                    ROOT = ROOT.parent

                SRC = ROOT / "src"
                if str(SRC) not in sys.path:
                    sys.path.insert(0, str(SRC))

                from project_name.infrastructure import (
                    INFRASTRUCTURE_DATASETS,
                    download_infrastructure_layers,
                )

                RAW_DIR = ROOT / "data" / "raw" / "infrastructure"
                PROCESSED_DIR = ROOT / "data" / "processed" / "infrastructure"
                NYC_BOUNDARY_PATH = ROOT / "data" / "spatial" / "vector" / "nyc_borough_boundary" / "nybb.geojson"

                RAW_DIR.mkdir(parents=True, exist_ok=True)
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                """
            ),
            md_cell(
                """
                ## Download

                Ejecuta la descarga una sola vez si quieres refrescar los archivos desde Socrata.
                """
            ),
            code_cell(
                """
                summary = download_infrastructure_layers(
                    raw_dir=RAW_DIR,
                    processed_dir=PROCESSED_DIR,
                    app_token=os.getenv("NYC_APP_TOKEN"),
                    limit=50_000,
                )

                summary_path = Path(summary.attrs["summary_path"])
                display(summary)
                print(f"Summary saved: {summary_path}")
                """
            ),
            md_cell(
                """
                ## Overview

                Visualiza las capas procesadas sobre el contorno de NYC cuando existe geometría puntual.
                """
            ),
            code_cell(
                """
                boundary = gpd.read_file(NYC_BOUNDARY_PATH).to_crs("EPSG:4326")

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                boundary.boundary.plot(ax=ax, color="black", linewidth=1)

                colors = {
                    "catch_basins": "#2563eb",
                    "green_infrastructure": "#16a34a",
                    "outfalls": "#dc2626",
                }

                for _, row in summary.iterrows():
                    if row["geometry_status"] != "saved":
                        continue
                    gdf = gpd.read_file(row["processed_path"])
                    if gdf.empty:
                        continue
                    gdf.plot(ax=ax, color=colors.get(row["layer_name"], "#6b7280"), markersize=5, alpha=0.8, label=row["layer_name"])

                ax.set_title("NYC infrastructure layers")
                ax.set_axis_off()
                ax.legend(frameon=False, loc="lower left")
                plt.show()
                """
            ),
            md_cell(
                """
                ## Uso

                - Si quieres correrlo como script, usa `scripts/download_infrastructure.py`.
                - Si alguna capa no trae coordenadas válidas, el notebook la reporta en `geometry_status`.
                - El resumen queda en `data/processed/infrastructure/infrastructure_download_summary.csv`.
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
