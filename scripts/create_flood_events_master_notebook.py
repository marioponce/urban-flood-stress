from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "flood-events-master-table.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Flood Event Master Table

This notebook builds the **event-level flood modeling dataset** for the Urban Flood Stress workflow.

Core rules:
- The base table is `data/processed/311/flood_events.csv`.
- The canonical street key is `segment_id`.
- The final unit of analysis is **Flood Event**, not individual 311 complaints.
- Precipitation is joined by `segment_id -> station_p_id` and temporal overlap.
- Tide is joined by `segment_id -> tide_id` and temporal overlap.
- Census, FEMA, and infrastructure are attached with spatial logic and event-safe aggregation.

Important temporal note:
- The current `311` event `end` timestamp is used as an **operational overlap window**.
- It is **not** treated as an observed closure timestamp unless the source status explicitly indicates closure.
- Therefore `resolution_bool` and `resolution_hours` are intentionally conservative."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.flood_events_master import (
    DIAGNOSTICS_DIR,
    MASTER_GEOPARQUET_PATH,
    MASTER_PARQUET_PATH,
    build_master_table,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """result = build_master_table(write_outputs=True)
master = result.master_table
diagnostics = result.diagnostics

print(f"rows: {len(master):,}")
print(f"unique events: {master['event_id'].nunique():,}")
print(f"unique segments: {master['segment_id'].nunique(dropna=True):,}")
print(f"master parquet: {MASTER_PARQUET_PATH}")
print(f"master geoparquet: {MASTER_GEOPARQUET_PATH}")
print(f"diagnostics dir: {DIAGNOSTICS_DIR}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Event Targets and Core Keys

The final target columns are:
- `occurrence`
- `intensity`
- `resolution_hours`
- `resolution_bool`

The event-level geometry comes from the matched `segment_id` line geometry in the LION-derived street layer."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(
    master[
        [
            "event_id",
            "segment_id",
            "event_start_local",
            "event_window_end_local",
            "event_window_duration_hours",
            "occurrence",
            "intensity",
            "resolution_hours",
            "resolution_bool",
            "station_p_id",
            "tide_id",
            "census_geoid",
            "fema_fld_zone",
        ]
    ].head(10)
)

display(diagnostics["validation_checks"])
display(diagnostics["precipitation_diagnostics"])
display(diagnostics["tide_diagnostics"])"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Join Quality and Missingness

These diagnostics help audit:
- missing joins,
- null inflation,
- precipitation and tide coverage,
- and whether one-row-per-event integrity was preserved."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(diagnostics["missingness_report"].head(40))
display(diagnostics["census_diagnostics"])
display(diagnostics["fema_diagnostics"])
display(diagnostics["infrastructure_diagnostics"])"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Target and Weather Diagnostics"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

master["intensity"].astype(float).plot.hist(ax=axes[0, 0], bins=30, color="#2563EB", alpha=0.85)
axes[0, 0].set_title("Flood Event Intensity")
axes[0, 0].set_xlabel("complaint count")

master["event_window_duration_hours"].astype(float).plot.hist(ax=axes[0, 1], bins=30, color="#059669", alpha=0.85)
axes[0, 1].set_title("Operational Event Duration")
axes[0, 1].set_xlabel("hours")

master["prec_depth_total"].astype(float).plot.hist(ax=axes[1, 0], bins=30, color="#7C3AED", alpha=0.85)
axes[1, 0].set_title("Overlapping Precipitation Depth")
axes[1, 0].set_xlabel("mm")

master["tide_level_m_max"].astype(float).plot.hist(ax=axes[1, 1], bins=30, color="#EA580C", alpha=0.85)
axes[1, 1].set_title("Maximum Tide Level During Event")
axes[1, 1].set_xlabel("m")

plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Spatial QA/QC"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """plot_frame = master
if not isinstance(plot_frame, gpd.GeoDataFrame):
    plot_frame = gpd.read_parquet(MASTER_GEOPARQUET_PATH)
    plot_frame = gpd.GeoDataFrame(plot_frame, geometry="geometry")

if plot_frame.crs is None:
    plot_frame = plot_frame.set_crs(2263)

plot_frame = plot_frame.to_crs(2263).copy()
plot_frame["zero_precip"] = plot_frame["n_prec"].eq(0)
plot_frame["intensity_num"] = pd.to_numeric(plot_frame["intensity"], errors="coerce").fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

plot_frame.plot(ax=axes[0], column="intensity_num", linewidth=1.0, legend=True, cmap="viridis")
axes[0].set_title("Event Segments by Intensity")
axes[0].set_axis_off()

plot_frame.plot(ax=axes[1], column="zero_precip", linewidth=1.0, legend=True, categorical=True, cmap="Set1")
axes[1].set_title("Flood Events with Zero Overlapping Precipitation")
axes[1].set_axis_off()

plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Saved Outputs

The build step writes:
- `data/processed/modeling/flood_events_master_table.parquet`
- `data/processed/modeling/flood_events_master_table.geoparquet`
- CSV diagnostics in `data/processed/modeling/diagnostics/`"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(master.head(5))
display(diagnostics["numeric_summary"].head(30))
display(diagnostics["correlation_summary"].head(30))"""
        )
    )

    nb["cells"] = cells
    NOTEBOOK_PATH.write_text(nbf.writes(nb))


if __name__ == "__main__":
    main()
