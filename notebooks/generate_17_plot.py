"""Generate notebooks/17_plot.ipynb — publication-ready visualization pipeline."""
import json, textwrap
from pathlib import Path

def md(source):
    return {"cell_type": "markdown", "id": None, "metadata": {}, "source": source}

def code(source, cell_id=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).lstrip("\n"),
    }

cells = []
_id = [0]
def next_id():
    _id[0] += 1
    return f"cell{_id[0]:08d}"

def add_md(source):
    c = md(source); c["id"] = next_id(); cells.append(c)

def add_code(source):
    c = code(source); c["id"] = next_id(); cells.append(c)

# ── Title ─────────────────────────────────────────────────────────────────────
add_md("""\
# 17 — Publication-Ready Visualizations

**Urban Flood Infrastructure Stress — NYC**

Narrative order: **Exploratory/QA-QC → Clustering → Occurrence + Intensity → Resolution → Diagnostics**

All figures saved to `figures/<section>/` as PNG (300 dpi) + PDF. A registry CSV is written to `tables/figures_inventory.csv`.""")

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
add_code("""\
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ── root ──────────────────────────────────────────────────────────────────────
_here = Path.cwd()
ROOT = _here.parent if _here.name == "notebooks" else _here

DATA   = ROOT / "data"
PROC   = DATA / "processed"
SPAT   = DATA / "spatial" / "vector"
TEMP   = DATA / "temporal"
MDIR   = PROC / "modeling"
FDIR   = ROOT / "figures"
TDIR   = ROOT / "tables"

# ── data paths ─────────────────────────────────────────────────────────────────
MASTER       = MDIR / "flood_events_master_table.parquet"
ARCHETYPES   = MDIR / "clustering_event_archetypes.parquet"
ARCHETYPES_GEO  = MDIR / "clustering_event_archetypes.geoparquet"
ANOMALIES    = MDIR / "anomaly_event_scores.parquet"
CLUST_SEL    = MDIR / "clustering_model_selection.csv"
OCC_PRED     = MDIR / "predictions_occurrence_ml.parquet"
INT_PRED     = MDIR / "predictions_intensity_ml.parquet"
RES_PRED     = MDIR / "predictions_resolution_ml.parquet"
OCC_RES      = MDIR / "results_occurrence_ml.csv"
INT_RES      = MDIR / "results_intensity_ml.csv"
RES_TIME_RES = MDIR / "results_resolution_time_ml.csv"
RES_CLOS_RES = MDIR / "results_resolution_closure_ml.csv"
FI_OCC       = MDIR / "feature_importance_occurrence.csv"
FI_INT       = MDIR / "feature_importance_intensity.csv"
FI_RES       = MDIR / "feature_importance_resolution.csv"
BIAS_OCC     = MDIR / "bias_diagnostics_occurrence.csv"
BIAS_RES     = MDIR / "bias_diagnostics_resolution.csv"
CALIB_OCC    = MDIR / "calibration_occurrence_ml.csv"

LION_METRICS = SPAT / "streets" / "processed" / "lion_metrics_elevation.gpkg"
LION_CONN    = SPAT / "streets" / "processed" / "lion_connectivity.gpkg"
BOROUGHS     = SPAT / "nyc_borough_boundary" / "nybb.geojson"
FEMA_ZONES   = SPAT / "fema_nfhl" / "nyc_nfhl_flood_zones.geojson"
TIDE_STA     = SPAT / "noaa" / "tide_station_points.geojson"
PREC_STA     = SPAT / "noaa" / "precipitation_station_points.geojson"
TIDE_SERIES  = TEMP / "noaa" / "tide_series.parquet"
PRECIP_EVT   = PROC / "precipitation" / "events_hourly" / "precipitation_events.csv"
MON_311      = TEMP / "311" / "311_monthly_borough_complaint_summary.csv"

# ── output dirs ────────────────────────────────────────────────────────────────
for _d in [
    FDIR / "exploratory" / "maps",
    FDIR / "exploratory" / "network",
    FDIR / "exploratory" / "hydroclimate",
    FDIR / "clustering",
    FDIR / "occurrence",
    FDIR / "intensity",
    FDIR / "resolution",
    FDIR / "diagnostics",
    TDIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ── palettes ──────────────────────────────────────────────────────────────────
BORO_PAL = {
    "Manhattan":   "#2563EB",
    "MANHATTAN":   "#2563EB",
    "Brooklyn":    "#16A34A",
    "BROOKLYN":    "#16A34A",
    "Queens":      "#D97706",
    "QUEENS":      "#D97706",
    "Bronx":       "#DC2626",
    "BRONX":       "#DC2626",
    "Staten Island": "#7C3AED",
    "STATEN ISLAND": "#7C3AED",
}
ADMIN_PAL = {
    "Bloomberg":  "#475569",
    "de Blasio":  "#2563EB",
    "Adams":      "#D97706",
    "Mamdani":    "#16A34A",
}

# ── figure registry ───────────────────────────────────────────────────────────
FIG_REGISTRY = []

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})
print("Setup complete. ROOT =", ROOT)
""")

# ── Cell 2: Helpers ───────────────────────────────────────────────────────────
add_code("""\
def safe_parquet(path, **kw):
    try:
        return pd.read_parquet(path, **kw)
    except Exception as e:
        print(f"  [skip] {Path(path).name}: {e}")
        return pd.DataFrame()

def safe_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        print(f"  [skip] {Path(path).name}: {e}")
        return pd.DataFrame()

def safe_gdf(path, layer=None):
    try:
        kw = {"layer": layer} if layer else {}
        return gpd.read_file(path, **kw)
    except Exception as e:
        print(f"  [skip] {Path(path).name}: {e}")
        return gpd.GeoDataFrame()

def to_proj(gdf, epsg=2263):
    if gdf is None or gdf.empty:
        return gdf
    try:
        return gdf.to_crs(epsg=epsg)
    except Exception:
        return gdf

def save_fig(fig, fig_id, subfolder=""):
    sub = FDIR / subfolder if subfolder else FDIR
    sub.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in [("png", 300), ("pdf", 300)]:
        p = sub / f"{fig_id}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(str(p))
    return paths

def reg(fig_id, section, title, paths, inputs=None, notes=""):
    FIG_REGISTRY.append({
        "figure_id": fig_id, "section": section, "title": title,
        "output_png": paths[0] if paths else "",
        "output_pdf": paths[1] if len(paths) > 1 else "",
        "input_files": "; ".join(str(f) for f in (inputs or [])),
        "created_successfully": bool(paths), "notes": notes,
    })
    print(f"  [saved] {fig_id}")

def col_ok(df, *cols):
    return all(c in df.columns for c in cols)

def qclip(s, lo=0.01, hi=0.99):
    v = s.quantile([lo, hi])
    return s.clip(v.iloc[0], v.iloc[1])

print("Helpers ready.")
""")

# ── Cell 3: Load data ─────────────────────────────────────────────────────────
add_code("""\
print("Loading datasets…")
master       = safe_parquet(MASTER)
archetypes   = safe_parquet(ARCHETYPES)
anomalies       = safe_parquet(ANOMALIES)
clust_sel    = safe_csv(CLUST_SEL)
occ_pred     = safe_parquet(OCC_PRED)
int_pred     = safe_parquet(INT_PRED)
res_pred     = safe_parquet(RES_PRED)
occ_res      = safe_csv(OCC_RES)
int_res      = safe_csv(INT_RES)
res_time_res = safe_csv(RES_TIME_RES)
res_clos_res = safe_csv(RES_CLOS_RES)
fi_occ       = safe_csv(FI_OCC)
fi_int       = safe_csv(FI_INT)
fi_res       = safe_csv(FI_RES)
bias_occ     = safe_csv(BIAS_OCC)
bias_res     = safe_csv(BIAS_RES)
calib_occ    = safe_csv(CALIB_OCC)
tide_series  = safe_parquet(TIDE_SERIES)
precip_evt   = safe_csv(PRECIP_EVT)
mon_311      = safe_csv(MON_311)

print(f"  master: {len(master):,} rows")
print(f"  archetypes: {len(archetypes):,} | anomalies: {len(anomalies):,}")
print(f"  occ_pred: {len(occ_pred):,} | int_pred: {len(int_pred):,} | res_pred: {len(res_pred):,}")
print(f"  tide_series: {len(tide_series):,} | precip_evt: {len(precip_evt):,}")

print("Loading spatial layers…")
boroughs  = to_proj(safe_gdf(BOROUGHS))
fema      = to_proj(safe_gdf(FEMA_ZONES))
tide_sta  = to_proj(safe_gdf(TIDE_STA))
prec_sta  = to_proj(safe_gdf(PREC_STA))
print(f"  boroughs: {len(boroughs)} | fema: {len(fema)} | tide_sta: {len(tide_sta)} | prec_sta: {len(prec_sta)}")

print("Loading LION (may take a moment)…")
lion = safe_gdf(LION_METRICS)
if not lion.empty:
    lion = to_proj(lion)
    print(f"  LION: {len(lion):,} segments, CRS={lion.crs}")
print("All data loaded.")
""")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXPLORATORY
# ════════════════════════════════════════════════════════════════════════════════
add_md("## 1 — Exploratory / QA-QC\n\nStudy area overview, network topology, hydroclimate signals, and complaint geography.")

# Fig 1.1 — Study area
add_code("""\
try:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()
    if not boroughs.empty:
        boroughs.plot(ax=ax, color="#F1F5F9", edgecolor="#94A3B8", linewidth=1.0, zorder=1)
    if not fema.empty and "FLD_ZONE" in fema.columns:
        sfha = fema[fema["FLD_ZONE"].str.startswith(("A", "V"), na=False)]
        if not sfha.empty:
            sfha.plot(ax=ax, color="#BFDBFE", alpha=0.55, linewidth=0, zorder=2)
    if not tide_sta.empty:
        tide_sta.plot(ax=ax, color="#1D4ED8", marker="^", markersize=80, zorder=5, label="Tide station")
    if not prec_sta.empty:
        prec_sta.plot(ax=ax, color="#059669", marker="s", markersize=65, zorder=5, label="Precip station")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Study Area — NYC Boroughs, FEMA Special Flood Hazard Areas, Monitoring Stations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_1_study_area", "exploratory/maps")
    reg("fig_1_1_study_area", "1-Exploratory", "Study area overview",
        paths, inputs=[BOROUGHS, FEMA_ZONES, TIDE_STA, PREC_STA])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.1: {exc}"); plt.close("all")
""")

# Fig 1.2 — Network betweenness
add_code("""\
try:
    if lion.empty or "edge_betweenness" not in lion.columns:
        raise ValueError("LION edge_betweenness unavailable")
    lp = lion.copy()
    lp["eb"] = qclip(lp["edge_betweenness"].fillna(0), 0.0, 0.99)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()
    lp.plot(ax=ax, column="eb", cmap="plasma", linewidth=0.35,
            legend=True, legend_kwds={"label": "Edge betweenness (99th-pctile clipped)", "shrink": 0.6},
            zorder=1)
    if not boroughs.empty:
        boroughs.plot(ax=ax, color="none", edgecolor="#CBD5E1", linewidth=1.2, zorder=2)
    ax.set_title("Street Network Edge Betweenness Centrality", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_2_network_betweenness", "exploratory/network")
    reg("fig_1_2_network_betweenness", "1-Exploratory", "Edge betweenness choropleth",
        paths, inputs=[LION_METRICS])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.2: {exc}"); plt.close("all")
""")

# Fig 1.3 — Terrain slope
add_code("""\
try:
    if lion.empty or "dem_slope" not in lion.columns:
        raise ValueError("LION dem_slope unavailable")
    ls = lion.copy()
    ls["slope"] = qclip(ls["dem_slope"].fillna(0), 0.0, 0.99)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()
    ls.plot(ax=ax, column="slope", cmap="RdYlBu_r", linewidth=0.35,
            legend=True, legend_kwds={"label": "Terrain slope ° (99th-pctile clipped)", "shrink": 0.6},
            zorder=1)
    if not boroughs.empty:
        boroughs.plot(ax=ax, color="none", edgecolor="#CBD5E1", linewidth=1.2, zorder=2)
    ax.set_title("Street Segment Terrain Slope from DEM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_3_terrain_slope", "exploratory/maps")
    reg("fig_1_3_terrain_slope", "1-Exploratory", "Terrain slope choropleth",
        paths, inputs=[LION_METRICS])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.3: {exc}"); plt.close("all")
""")

# Fig 1.4 — Tide time series
add_code("""\
try:
    if tide_series.empty:
        raise ValueError("Tide series unavailable")
    ts = tide_series.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts = ts.sort_values("timestamp")
    if "role" in ts.columns:
        ts = ts[ts["role"].isin(["primary", "reference", "secondary"])]
    if "tide_level_ft" not in ts.columns:
        raise ValueError("tide_level_ft column missing")
    ts = ts.set_index("timestamp")
    monthly = (
        ts.groupby("station_name")["tide_level_ft"]
        .resample("ME").mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.get_cmap("tab10", monthly["station_name"].nunique())
    for i, (sta, grp) in enumerate(monthly.groupby("station_name")):
        ax.plot(grp["timestamp"], grp["tide_level_ft"], label=sta,
                color=colors(i), linewidth=1.1, alpha=0.9)
    ax.set_xlabel("Date"); ax.set_ylabel("Mean Monthly Tide Level (ft)")
    ax.set_title("Tidal Observations — NOAA Stations (Monthly Mean)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_4_tide_series", "exploratory/hydroclimate")
    reg("fig_1_4_tide_series", "1-Exploratory", "Tide time series", paths, inputs=[TIDE_SERIES])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.4: {exc}"); plt.close("all")
""")

# Fig 1.5 — Precipitation climatology
add_code("""\
try:
    if precip_evt.empty:
        raise ValueError("Precip events unavailable")
    pe = precip_evt.copy()
    if "start" in pe.columns:
        pe["start"] = pd.to_datetime(pe["start"])
        pe["month"] = pe["start"].dt.month
    else:
        raise ValueError("No start column")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "depth" in pe.columns:
        md = pe.groupby("month")["depth"].mean()
        axes[0].bar(md.index, md.values, color="#2563EB", alpha=0.82, edgecolor="white")
        axes[0].set_xlabel("Month"); axes[0].set_ylabel("Mean Event Depth (mm)")
        axes[0].set_title("Mean Precipitation Event Depth by Month")
        axes[0].set_xticks(range(1, 13))
    mc = pe.groupby("month").size()
    axes[1].bar(mc.index, mc.values, color="#059669", alpha=0.82, edgecolor="white")
    axes[1].set_xlabel("Month"); axes[1].set_ylabel("Number of Events")
    axes[1].set_title("Precipitation Event Count by Month")
    axes[1].set_xticks(range(1, 13))
    plt.suptitle("Precipitation Climatology — NYC ASOS Stations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_5_precip_climatology", "exploratory/hydroclimate")
    reg("fig_1_5_precip_climatology", "1-Exploratory", "Precipitation climatology",
        paths, inputs=[PRECIP_EVT])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.5: {exc}"); plt.close("all")
""")

# Fig 1.6 — Complaint bubble map
add_code("""\
try:
    if master.empty or not col_ok(master, "event_lat", "event_lon", "n_complaints"):
        raise ValueError("Master table missing event_lat/event_lon/n_complaints")
    m = master[["event_lat", "event_lon", "n_complaints"]].dropna()
    gdf_pts = gpd.GeoDataFrame(
        m, geometry=gpd.points_from_xy(m["event_lon"], m["event_lat"]), crs="EPSG:4326"
    )
    gdf_pts = to_proj(gdf_pts)
    sz = qclip(gdf_pts["n_complaints"], 0.0, 0.98)
    sz_norm = (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()
    if not boroughs.empty:
        boroughs.plot(ax=ax, color="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2, zorder=1)
    sc = ax.scatter(
        gdf_pts.geometry.x, gdf_pts.geometry.y,
        c=gdf_pts["n_complaints"], cmap="YlOrRd",
        s=(sz_norm * 55 + 2).clip(1, 60), alpha=0.45, linewidths=0, zorder=2
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Complaints per event")
    ax.set_title("Flood Complaint Locations Sized by Event Complaint Volume",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_6_complaint_bubble_map", "exploratory/maps")
    reg("fig_1_6_complaint_bubble_map", "1-Exploratory", "Complaint bubble map",
        paths, inputs=[MASTER])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.6: {exc}"); plt.close("all")
""")

# Fig 1.7 — Monthly 311 trends
add_code("""\
try:
    if mon_311.empty:
        raise ValueError("Monthly 311 summary unavailable")
    m3 = mon_311.copy()
    date_col = next((c for c in ["created_month", "month", "date", "period"] if c in m3.columns), None)
    if date_col is None:
        raise ValueError("No date column in monthly 311 data")
    m3["created_month"] = pd.to_datetime(m3[date_col])
    fig, ax = plt.subplots(figsize=(14, 5))
    req_col = next((c for c in ["requests", "n_complaints", "count"] if c in m3.columns), None)
    if req_col and "borough" in m3.columns:
        for boro, grp in m3.groupby("borough"):
            monthly = grp.groupby("created_month")[req_col].sum()
            ax.plot(monthly.index, monthly.values,
                    label=boro, color=BORO_PAL.get(boro, "#64748B"), linewidth=1.0, alpha=0.85)
    elif req_col:
        monthly = m3.groupby("created_month")[req_col].sum()
        ax.plot(monthly.index, monthly.values, color="#2563EB", linewidth=1.2)
    else:
        raise ValueError("No count column found")
    # shade administrations
    admin_spans = [
        ("Bloomberg", "2002-01", "2013-12", "#94A3B8"),
        ("de Blasio", "2014-01", "2021-12", "#BFDBFE"),
        ("Adams",     "2022-01", "2025-02", "#FEF3C7"),
        ("Mamdani",   "2025-03", "2026-06", "#D1FAE5"),
    ]
    for label, t0, t1, col in admin_spans:
        ts0, ts1 = pd.Timestamp(t0), pd.Timestamp(t1)
        if ts0 <= m3["created_month"].max() and ts1 >= m3["created_month"].min():
            ax.axvspan(ts0, ts1, alpha=0.13, color=col, zorder=0)
            mid = ts0 + (ts1 - ts0) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.92, label, ha="center",
                    fontsize=8, color="#475569")
    ax.set_xlabel("Date"); ax.set_ylabel("Monthly Flood 311 Requests")
    ax.set_title("Monthly 311 Flood Complaints by Borough and Mayoral Administration",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_7_monthly_311_trends", "exploratory/hydroclimate")
    reg("fig_1_7_monthly_311_trends", "1-Exploratory", "Monthly 311 complaint trends",
        paths, inputs=[MON_311])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.7: {exc}"); plt.close("all")
""")

# Fig 1.8 — Event duration
add_code("""\
try:
    if master.empty or "duration_hours" not in master.columns:
        raise ValueError("duration_hours missing")
    dur = master["duration_hours"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(qclip(dur, 0, 0.99), bins=50, color="#2563EB", alpha=0.82, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Event Duration (hours)"); axes[0].set_ylabel("Events")
    axes[0].set_title("Distribution of Flood Event Durations")
    if col_ok(master, "duration_hours", "n_complaints"):
        sub = master[["duration_hours", "n_complaints"]].dropna()
        axes[1].scatter(
            qclip(sub["duration_hours"], 0, 0.99),
            qclip(sub["n_complaints"], 0, 0.99),
            alpha=0.15, s=8, color="#7C3AED", linewidths=0
        )
        axes[1].set_xlabel("Event Duration (hours)"); axes[1].set_ylabel("Complaints per Event")
        axes[1].set_title("Event Duration vs. Complaint Volume")
    plt.suptitle("Flood Event Characteristics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_1_8_event_duration", "exploratory/hydroclimate")
    reg("fig_1_8_event_duration", "1-Exploratory", "Event duration distribution",
        paths, inputs=[MASTER])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 1.8: {exc}"); plt.close("all")
""")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════
add_md("## 2 — Clustering\n\nEvent archetypes from combined feature space (K-Means, GMM, Agglomerative). Physical-forcing clusters achieve the highest silhouette score.")

# Fig 2.1 — Clustering model selection
add_code("""\
try:
    if clust_sel.empty:
        raise ValueError("clustering_model_selection unavailable")
    cs = clust_sel.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, fs in zip(axes, cs["feature_set"].unique()[:2]):
        grp = cs[cs["feature_set"] == fs].sort_values("k")
        for alg, sub in grp.groupby("algorithm"):
            ax.plot(sub["k"], sub["silhouette_score"], marker="o", markersize=5,
                    linewidth=1.2, label=alg)
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title(f"Feature set: {fs}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Clustering Algorithm Selection — Silhouette Score vs. k",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_2_1_cluster_selection", "clustering")
    reg("fig_2_1_cluster_selection", "2-Clustering", "Clustering model selection",
        paths, inputs=[CLUST_SEL])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 2.1: {exc}"); plt.close("all")
""")

# Fig 2.2 — PCA projection
add_code("""\
try:
    if archetypes.empty:
        raise ValueError("Archetypes unavailable")
    feat_cols = [c for c in ["max_tide", "prec_depth_total", "elevation",
                              "shore_dist", "travel_time"] if c in archetypes.columns]
    if len(feat_cols) < 2:
        raise ValueError(f"Insufficient feature cols: {feat_cols}")
    X = archetypes[feat_cols].fillna(0).to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(StandardScaler().fit_transform(X))
    clabels = archetypes["combined_cluster_id"].astype(str).values if "combined_cluster_id" in archetypes.columns else ["0"] * len(archetypes)
    unique_c = sorted(set(clabels))
    cmap = plt.get_cmap("tab10", len(unique_c))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cid in enumerate(unique_c):
        mask = clabels == cid
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=12, alpha=0.55,
                   color=cmap(i), label=f"Cluster {cid}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA Projection of Flood Events Colored by Cluster Assignment",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", markerscale=2, fontsize=8)
    plt.tight_layout()
    paths = save_fig(fig, "fig_2_2_pca_clusters", "clustering")
    reg("fig_2_2_pca_clusters", "2-Clustering", "PCA projection by cluster",
        paths, inputs=[ARCHETYPES])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 2.2: {exc}"); plt.close("all")
""")

# Fig 2.3 — Spatial segment cluster map  (uses geoparquet — has geometry + cluster already)
add_code("""\
try:
    # Load geoparquet which already has segment geometry + combined_cluster_id
    arc_geo = gpd.read_parquet(ARCHETYPES_GEO)
    if arc_geo.empty or "combined_cluster_id" not in arc_geo.columns:
        raise ValueError("Archetypes geoparquet unavailable or missing combined_cluster_id")
    if arc_geo.crs is None:
        arc_geo = arc_geo.set_crs(epsg=2263)
    else:
        arc_geo = to_proj(arc_geo)

    # Mode cluster per segment — keep one representative geometry per segment
    def _mode(x):
        m = x.mode(); return m.iloc[0] if len(m) else pd.NA

    seg_gdf = (
        arc_geo.dissolve(by="segment_id", aggfunc={"combined_cluster_id": _mode})
        .reset_index()
        .rename(columns={"combined_cluster_id": "dominant_cluster"})
    )

    n_matched = seg_gdf["dominant_cluster"].notna().sum()
    n_clusters = seg_gdf["dominant_cluster"].dropna().nunique()
    print(f"  {n_matched:,} / {len(seg_gdf):,} segments with cluster assignment ({n_clusters} clusters)")

    seg_gdf["dominant_cluster"] = seg_gdf["dominant_cluster"].astype(str)
    cmap_name = "tab10" if n_clusters <= 10 else "tab20"

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_axis_off()  # MUST precede geo.plot() on EPSG:2263

    seg_gdf.plot(ax=ax, column="dominant_cluster", categorical=True,
                 cmap=cmap_name, linewidth=0.7, alpha=0.85,
                 legend=True,
                 legend_kwds={"title": "Dominant Cluster", "loc": "lower right",
                              "fontsize": 9, "title_fontsize": 10},
                 zorder=1)
    if not boroughs.empty:
        boroughs.plot(ax=ax, color="none", edgecolor="#334155", linewidth=1.2, zorder=2)

    ax.set_title("NYC Street Segments Colored by Dominant Flood Event Cluster Assignment",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_2_3_spatial_segment_clusters", "clustering")
    reg("fig_2_3_spatial_segment_clusters", "2-Clustering",
        "Spatial segment cluster map",
        paths, inputs=[ARCHETYPES_GEO],
        notes=f"{n_matched:,} matched segments, {n_clusters} clusters")
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 2.3: {exc}"); plt.close("all")
""")

# Fig 2.4 — Cluster profiles
add_code("""\
try:
    if archetypes.empty or "combined_cluster_id" not in archetypes.columns:
        raise ValueError("archetypes missing combined_cluster_id")
    profile_cols = [c for c in ["max_tide", "prec_depth_total", "elevation",
                                  "shore_dist", "travel_time", "intensity", "resolution"]
                    if c in archetypes.columns]
    if not profile_cols:
        raise ValueError("No profile cols available")
    arc = archetypes.copy()
    arc["combined_cluster_id"] = arc["combined_cluster_id"].astype(str)
    cp = arc.groupby("combined_cluster_id")[profile_cols].mean(numeric_only=True)
    cnt = arc["combined_cluster_id"].value_counts().sort_index()
    cp_z = (cp - cp.mean()) / (cp.std() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cnt.plot.bar(ax=axes[0], color="#2563EB", alpha=0.85, edgecolor="white")
    axes[0].set_title("Cluster Event Counts")
    axes[0].set_xlabel("Cluster ID"); axes[0].set_ylabel("Events")
    axes[0].tick_params(axis="x", rotation=0)

    im = axes[1].imshow(cp_z.to_numpy(dtype=float), aspect="auto", cmap="RdYlBu_r")
    axes[1].set_title("Cluster Feature Profiles (z-scored)")
    axes[1].set_xticks(range(len(profile_cols)))
    axes[1].set_xticklabels(profile_cols, rotation=45, ha="right")
    axes[1].set_yticks(range(len(cp.index)))
    axes[1].set_yticklabels(cp.index.astype(str))
    plt.colorbar(im, ax=axes[1], shrink=0.8, label="z-score")
    plt.suptitle("Flood Event Cluster Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_2_4_cluster_profiles", "clustering")
    reg("fig_2_4_cluster_profiles", "2-Clustering", "Cluster profiles heatmap",
        paths, inputs=[ARCHETYPES])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 2.4: {exc}"); plt.close("all")
""")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — OCCURRENCE + INTENSITY
# ════════════════════════════════════════════════════════════════════════════════
add_md("## 3 — Occurrence and Intensity\n\nClassification performance (ROC, confusion matrix, feature importance) and regression diagnostics for complaint intensity.")

# Fig 3.1 — ROC curves
add_code("""\
try:
    if occ_pred.empty:
        raise ValueError("Occurrence predictions unavailable")
    test_set = occ_pred[occ_pred["set_name"] == "test"].copy()
    if test_set.empty:
        raise ValueError("No test rows in occurrence predictions")
    models = test_set["model_name"].unique()
    cmap = plt.get_cmap("tab10", len(models))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, model in enumerate(models):
        sub = test_set[test_set["model_name"] == model].dropna(subset=["y_true"])
        score_col = "y_score" if ("y_score" in sub.columns and sub["y_score"].notna().any()) else None
        if score_col:
            valid = sub.dropna(subset=["y_score"])
        else:
            valid = sub
            score_col = "y_pred"
        if len(valid) < 10:
            continue
        fpr, tpr, _ = roc_curve(valid["y_true"].astype(int), valid[score_col].astype(float))
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cmap(i), linewidth=1.5,
                label=f"{model.replace('_',' ')} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random baseline")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Flood Occurrence Classification (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    paths = save_fig(fig, "fig_3_1_roc_curves", "occurrence")
    reg("fig_3_1_roc_curves", "3-Occurrence", "ROC curves", paths, inputs=[OCC_PRED])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 3.1: {exc}"); plt.close("all")
""")

# Fig 3.2 — Feature importance (occurrence)
add_code("""\
try:
    if fi_occ.empty or "feature" not in fi_occ.columns:
        raise ValueError("Feature importance occurrence unavailable")
    fi = fi_occ.dropna(subset=["importance"]).sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563EB"] * len(fi)
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Gini Importance")
    ax.set_title("Top 15 Feature Importances — Occurrence (Random Forest)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    paths = save_fig(fig, "fig_3_2_feature_importance_occurrence", "occurrence")
    reg("fig_3_2_feature_importance_occurrence", "3-Occurrence",
        "Feature importance (occurrence RF)", paths, inputs=[FI_OCC])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 3.2: {exc}"); plt.close("all")
""")

# Fig 3.3 — Confusion matrix + calibration
add_code("""\
try:
    if occ_pred.empty:
        raise ValueError("Occurrence predictions unavailable")
    # Best model = random_forest on test set
    sub = occ_pred[
        (occ_pred["set_name"] == "test") &
        (occ_pred["model_name"] == "random_forest")
    ].dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        sub = occ_pred[occ_pred["set_name"] == "test"].dropna(subset=["y_true", "y_pred"])
    cm = confusion_matrix(sub["y_true"].astype(int), sub["y_pred"].astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # confusion matrix
    im = axes[0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Pred: No flood", "Pred: Flood"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["True: No flood", "True: Flood"])
    for r in range(2):
        for c in range(2):
            axes[0].text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                         fontsize=13, color="white" if cm[r, c] > cm.max() / 2 else "black")
    axes[0].set_title("Confusion Matrix — Random Forest (Test Set)")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # calibration curve
    if not calib_occ.empty and col_ok(calib_occ, "mean_predicted_probability", "observed_rate"):
        calib_rf = calib_occ[
            (calib_occ["model_name"].str.contains("random_forest", na=False)) &
            (calib_occ["set_name"] == "test")
        ]
        if calib_rf.empty:
            calib_rf = calib_occ[calib_occ["set_name"] == "test"]
        if not calib_rf.empty:
            axes[1].scatter(calib_rf["mean_predicted_probability"],
                            calib_rf["observed_rate"], s=40, alpha=0.75, color="#2563EB")
            axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="Perfect calibration")
            axes[1].set_xlabel("Mean Predicted Probability")
            axes[1].set_ylabel("Observed Positive Rate")
            axes[1].set_title("Calibration Curve — Occurrence (Test Set)")
            axes[1].legend(); axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Calibration data unavailable",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=12)

    plt.suptitle("Occurrence Model Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_3_3_confusion_calibration", "occurrence")
    reg("fig_3_3_confusion_calibration", "3-Occurrence",
        "Confusion matrix + calibration", paths, inputs=[OCC_PRED, CALIB_OCC])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 3.3: {exc}"); plt.close("all")
""")

# Fig 3.4 — Borough error rates (occurrence bias)
add_code("""\
try:
    if bias_occ.empty:
        raise ValueError("Occurrence bias diagnostics unavailable")
    bo = bias_occ[bias_occ["bias_dimension"] == "borough"].copy() if "bias_dimension" in bias_occ.columns else bias_occ.copy()
    if "group_value" not in bo.columns or "error_rate" not in bo.columns:
        raise ValueError("Expected bias columns missing")
    bo = bo.sort_values("error_rate", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["error_rate", "false_negative_rate", "false_positive_rate"]
    titles = ["Error Rate", "False Negative Rate", "False Positive Rate"]
    for ax, met, ttl in zip(axes, metrics, titles):
        if met not in bo.columns:
            ax.text(0.5, 0.5, f"{met} unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        colors = [BORO_PAL.get(b, "#64748B") for b in bo["group_value"]]
        ax.bar(bo["group_value"], bo[met], color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Borough"); ax.set_ylabel(ttl)
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Occurrence Model Bias by Borough (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_3_4_occurrence_bias_borough", "occurrence")
    reg("fig_3_4_occurrence_bias_borough", "3-Occurrence",
        "Occurrence bias by borough", paths, inputs=[BIAS_OCC])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 3.4: {exc}"); plt.close("all")
""")

# Fig 3.5 — Intensity predicted vs actual
add_code("""\
try:
    if int_pred.empty:
        raise ValueError("Intensity predictions unavailable")
    test_i = int_pred[int_pred["set_name"] == "test"].dropna(subset=["y_true", "y_pred"])
    if test_i.empty:
        raise ValueError("No test rows in intensity predictions")
    # best model by test r2
    best_int_model = None
    if not int_res.empty and "model_name" in int_res.columns and "test_r2" in int_res.columns:
        best_int_model = int_res.dropna(subset=["test_r2"]).sort_values("test_r2", ascending=False).iloc[0]["model_name"]
    if best_int_model and best_int_model in test_i["model_name"].values:
        sub_i = test_i[test_i["model_name"] == best_int_model]
    else:
        sub_i = test_i[test_i["model_name"] == test_i["model_name"].iloc[0]]
        best_int_model = sub_i["model_name"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(sub_i["y_true"], sub_i["y_pred"],
                    alpha=0.2, s=8, color="#7C3AED", linewidths=0)
    lim = max(sub_i["y_true"].max(), sub_i["y_pred"].max())
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6, label="y = x")
    axes[0].set_xlabel("True Intensity (log1p complaints)"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Intensity: Predicted vs. Actual\\n({best_int_model}, test set)")
    axes[0].legend()

    # R² per model bar chart
    if not int_res.empty and "test_r2" in int_res.columns and "model_name" in int_res.columns:
        best_split = "temporal_by_administration"
        ir = int_res[int_res["split_strategy"] == best_split] if "split_strategy" in int_res.columns else int_res
        if ir.empty:
            ir = int_res
        ir_sorted = ir.sort_values("test_r2")
        axes[1].barh(ir_sorted["model_name"].str.replace("_", " "),
                     ir_sorted["test_r2"], color="#7C3AED", alpha=0.82, edgecolor="white")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Test R²")
        axes[1].set_title("Intensity Regression R² by Model (Temporal Split)")
        axes[1].grid(axis="x", alpha=0.3)
    plt.suptitle("Complaint Intensity — Regression Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_3_5_intensity_diagnostics", "intensity")
    reg("fig_3_5_intensity_diagnostics", "3-Intensity", "Intensity predicted vs actual",
        paths, inputs=[INT_PRED, INT_RES])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 3.5: {exc}"); plt.close("all")
""")

# Fig 3.6 — Intensity residuals by borough
add_code("""\
try:
    if int_pred.empty or not col_ok(int_pred, "y_true", "y_pred", "borough"):
        raise ValueError("Intensity predictions or borough column unavailable")
    test_i = int_pred[int_pred["set_name"] == "test"].dropna(subset=["y_true", "y_pred", "borough"])
    if test_i.empty:
        raise ValueError("No test rows")
    test_i = test_i.copy()
    test_i["residual"] = test_i["y_pred"] - test_i["y_true"]
    boros = sorted(test_i["borough"].unique())
    data = [test_i.loc[test_i["borough"] == b, "residual"].dropna().values for b in boros]
    colors = [BORO_PAL.get(b, BORO_PAL.get(b.title(), "#64748B")) for b in boros]
    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(data, positions=range(len(boros)), showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(boros))); ax.set_xticklabels(boros, rotation=15)
    ax.set_ylabel("Prediction Residual (log1p scale)")
    ax.set_title("Intensity Prediction Residuals by Borough (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    paths = save_fig(fig, "fig_3_6_intensity_residuals_borough", "intensity")
    reg("fig_3_6_intensity_residuals_borough", "3-Intensity",
        "Intensity residuals by borough", paths, inputs=[INT_PRED])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 3.6: {exc}"); plt.close("all")
""")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESOLUTION
# ════════════════════════════════════════════════════════════════════════════════
add_md("## 4 — Resolution\n\nResolution time regression and closure classification. Tidal forcing is the dominant predictor; Queens exhibits a ninefold MAE anomaly relative to the Bronx.")

# Fig 4.1 — Resolution predicted vs actual
add_code("""\
try:
    if res_pred.empty:
        raise ValueError("Resolution predictions unavailable")
    rt = res_pred[
        (res_pred["target"] == "resolution") &
        (res_pred["set_name"] == "test")
    ].dropna(subset=["y_true", "y_pred"])
    if rt.empty:
        raise ValueError("No resolution time test predictions")
    best_model = "random_forest_regressor"
    if best_model not in rt["model_name"].values:
        best_model = rt["model_name"].iloc[0]
    sub = rt[rt["model_name"] == best_model]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(sub["y_true"], sub["y_pred"], alpha=0.2, s=8,
                    color="#D97706", linewidths=0)
    lim = max(sub["y_true"].max(), sub["y_pred"].max())
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6, label="y = x")
    axes[0].set_xlabel("True Resolution Time (log1p hours)")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Resolution Time: Predicted vs. Actual\\n({best_model}, test set)")
    axes[0].legend()

    # R² per model
    if not res_time_res.empty and "test_r2" in res_time_res.columns and "model_name" in res_time_res.columns:
        rr = res_time_res
        if "split_strategy" in rr.columns:
            rr = rr[rr["split_strategy"] == "temporal_by_administration"]
        if rr.empty:
            rr = res_time_res
        rr_s = rr.sort_values("test_r2")
        axes[1].barh(rr_s["model_name"].str.replace("_", " "),
                     rr_s["test_r2"], color="#D97706", alpha=0.82, edgecolor="white")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Test R²")
        axes[1].set_title("Resolution Time R² by Model (Temporal Split)")
        axes[1].grid(axis="x", alpha=0.3)
    plt.suptitle("Resolution Time Regression Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_4_1_resolution_diagnostics", "resolution")
    reg("fig_4_1_resolution_diagnostics", "4-Resolution", "Resolution time diagnostics",
        paths, inputs=[RES_PRED, RES_TIME_RES])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 4.1: {exc}"); plt.close("all")
""")

# Fig 4.2 — Feature importance (resolution)
add_code("""\
try:
    if fi_res.empty or "feature" not in fi_res.columns:
        raise ValueError("Resolution feature importance unavailable")
    # prefer resolution_time_regression subtask if available
    if "subtask" in fi_res.columns:
        fi_time = fi_res[fi_res["subtask"].str.contains("time", na=False)].dropna(subset=["importance"])
        if fi_time.empty:
            fi_time = fi_res.dropna(subset=["importance"])
    else:
        fi_time = fi_res.dropna(subset=["importance"])
    fi_top = fi_time.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_top["feature"][::-1], fi_top["importance"][::-1],
            color="#D97706", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances — Resolution Time (Random Forest Regressor)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    paths = save_fig(fig, "fig_4_2_feature_importance_resolution", "resolution")
    reg("fig_4_2_feature_importance_resolution", "4-Resolution",
        "Feature importance (resolution RFR)", paths, inputs=[FI_RES])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 4.2: {exc}"); plt.close("all")
""")

# Fig 4.3 — Borough MAE (resolution bias)
add_code("""\
try:
    if bias_res.empty or not col_ok(bias_res, "group_value", "mae"):
        raise ValueError("Resolution bias diagnostics unavailable")
    br = bias_res.copy()
    if "bias_dimension" in br.columns:
        br = br[br["bias_dimension"] == "borough"]
    if "task_name" in br.columns:
        br = br[br["task_name"] == "resolution_time"]
    if "set_name" in br.columns:
        br = br[br["set_name"] == "test"]
    br = br.sort_values("mae", ascending=False)
    colors = [BORO_PAL.get(b, BORO_PAL.get(b.title(), "#64748B")) for b in br["group_value"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(br["group_value"], br["mae"], color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Borough"); axes[0].set_ylabel("Mean Absolute Error (log1p hours)")
    axes[0].set_title("Resolution Time MAE by Borough")
    axes[0].tick_params(axis="x", rotation=15); axes[0].grid(axis="y", alpha=0.3)
    # annotate Queens outlier
    if "QUEENS" in br["group_value"].values or "Queens" in br["group_value"].values:
        qv = br[br["group_value"].isin(["QUEENS", "Queens"])]["mae"].values
        if len(qv):
            axes[0].annotate(f"Queens\\nMAE={qv[0]:.0f}",
                             xy=(br[br["group_value"].isin(["QUEENS","Queens"])].index[0] if False else 0, qv[0]),
                             xytext=(0, qv[0] * 1.05),
                             ha="center", fontsize=9, color="#DC2626")
    if "underprediction_rate" in br.columns:
        axes[1].bar(br["group_value"], br["underprediction_rate"],
                    color=colors, alpha=0.85, edgecolor="white")
        axes[1].set_xlabel("Borough"); axes[1].set_ylabel("Underprediction Rate")
        axes[1].set_title("Resolution Time Underprediction Rate by Borough")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].set_ylim(0, 1); axes[1].grid(axis="y", alpha=0.3)
    elif "mean_residual" in br.columns:
        axes[1].bar(br["group_value"], br["mean_residual"], color=colors, alpha=0.85, edgecolor="white")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Borough"); axes[1].set_ylabel("Mean Residual")
        axes[1].set_title("Mean Residual by Borough (+ = overprediction)")
        axes[1].tick_params(axis="x", rotation=15); axes[1].grid(axis="y", alpha=0.3)
    plt.suptitle("Resolution Time Bias by Borough (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_4_3_resolution_bias_borough", "resolution")
    reg("fig_4_3_resolution_bias_borough", "4-Resolution",
        "Resolution time bias by borough", paths, inputs=[BIAS_RES])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 4.3: {exc}"); plt.close("all")
""")

# Fig 4.4 — Resolution by administration (violin)
add_code("""\
try:
    if res_pred.empty or not col_ok(res_pred, "y_true", "mayoral_administration"):
        raise ValueError("Resolution predictions or mayoral_administration unavailable")
    rt = res_pred[
        (res_pred["target"] == "resolution") &
        (res_pred["model_name"] == "random_forest_regressor")
    ].dropna(subset=["y_true", "mayoral_administration"])
    if rt.empty:
        rt = res_pred[res_pred["target"] == "resolution"].dropna(subset=["y_true", "mayoral_administration"])
    if rt.empty:
        raise ValueError("No rows for resolution violin plot")
    admins = [a for a in ["Bloomberg", "de Blasio", "Adams", "Mamdani"]
              if a in rt["mayoral_administration"].values]
    if not admins:
        admins = rt["mayoral_administration"].unique()
    data = [rt.loc[rt["mayoral_administration"] == a, "y_true"].dropna().values for a in admins]
    data = [d for d in data if len(d) > 10]
    if not data:
        raise ValueError("Insufficient data for violin")
    colors = [ADMIN_PAL.get(a, "#64748B") for a in admins[:len(data)]]
    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(admins[:len(data)])
    ax.set_ylabel("True Resolution Time (log1p hours)")
    ax.set_title("Flood Complaint Resolution Time by Mayoral Administration",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    paths = save_fig(fig, "fig_4_4_resolution_by_administration", "resolution")
    reg("fig_4_4_resolution_by_administration", "4-Resolution",
        "Resolution time by administration", paths, inputs=[RES_PRED])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 4.4: {exc}"); plt.close("all")
""")

# Fig 4.5 — Resolution spatial map
add_code("""\
try:
    if res_pred.empty or master.empty or lion.empty:
        raise ValueError("res_pred/master/lion required for spatial map")
    if not col_ok(res_pred, "event_id", "y_true") or not col_ok(master, "event_id", "segment_id"):
        raise ValueError("Join columns missing")

    rt_all = res_pred[res_pred["target"] == "resolution"].dropna(subset=["event_id", "y_true"])
    # median resolution per event
    evt_res = rt_all.groupby("event_id")["y_true"].median().reset_index(name="median_res")
    # join to segment_id via master
    seg_res = (
        master[["event_id", "segment_id"]]
        .merge(evt_res, on="event_id", how="inner")
    )
    seg_med = (
        seg_res.groupby("segment_id")["median_res"]
        .median()
        .reset_index()
    )

    lion_r = lion.copy()
    lion_r["_key"] = lion_r["SegmentID"].astype(str).str.strip()
    seg_med["_key"] = seg_med["segment_id"].astype(str).str.strip()
    lion_r = lion_r.merge(seg_med[["_key", "median_res"]], on="_key", how="left")

    n_matched = lion_r["median_res"].notna().sum()
    print(f"  Resolution spatial: {n_matched:,} matched segments")

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_axis_off()

    unmatched = lion_r[lion_r["median_res"].isna()]
    if not unmatched.empty:
        unmatched.plot(ax=ax, color="#DDE3EC", linewidth=0.22, alpha=0.38, zorder=1)

    matched = lion_r.dropna(subset=["median_res"])
    if not matched.empty:
        matched.plot(ax=ax, column="median_res", cmap="YlOrRd", linewidth=0.5, alpha=0.85,
                     legend=True,
                     legend_kwds={"label": "Median Resolution Time (log1p hours)", "shrink": 0.6},
                     zorder=2)
    if not boroughs.empty:
        boroughs.plot(ax=ax, color="none", edgecolor="#334155", linewidth=1.2, zorder=3)

    ax.set_title("Median Flood Complaint Resolution Time by Street Segment",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_4_5_resolution_spatial_map", "resolution")
    reg("fig_4_5_resolution_spatial_map", "4-Resolution",
        "Resolution time spatial map", paths,
        inputs=[RES_PRED, MASTER, LION_METRICS],
        notes=f"{n_matched:,} matched segments")
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 4.5: {exc}"); plt.close("all")
""")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════════
add_md("## 5 — Diagnostics\n\nCross-task model metrics summary, anomaly detection, and fairness audit by poverty tercile.")

# Fig 5.1 — Model metrics heatmap
add_code("""\
try:
    rows = []
    if not occ_res.empty and "model_name" in occ_res.columns:
        for _, r in occ_res.iterrows():
            sp = r.get("split_strategy", "")
            if "temporal" in str(sp):
                rows.append({"task": "occurrence", "model": r["model_name"],
                             "metric": "ROC-AUC", "value": r.get("test_roc_auc", np.nan)})
                rows.append({"task": "occurrence", "model": r["model_name"],
                             "metric": "F1", "value": r.get("test_f1", np.nan)})
    if not res_time_res.empty and "model_name" in res_time_res.columns:
        for _, r in res_time_res.iterrows():
            sp = r.get("split_strategy", "")
            if "temporal" in str(sp):
                rows.append({"task": "resolution", "model": r["model_name"],
                             "metric": "R²", "value": r.get("test_r2", np.nan)})
                rows.append({"task": "resolution", "model": r["model_name"],
                             "metric": "MAE", "value": r.get("test_mae", np.nan)})
    if not rows:
        raise ValueError("No metrics to display")

    metrics_df = pd.DataFrame(rows)
    pivot = metrics_df.pivot_table(index=["task", "model"], columns="metric", values="value")
    pivot_clean = pivot.dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot_clean) * 0.5)))
    vmin, vmax = pivot_clean.min().min(), pivot_clean.max().max()
    im = ax.imshow(pivot_clean.to_numpy(dtype=float), aspect="auto",
                   cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot_clean.columns)))
    ax.set_xticklabels(pivot_clean.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_clean.index)))
    ax.set_yticklabels([f"{t} | {m}" for t, m in pivot_clean.index])
    plt.colorbar(im, ax=ax, shrink=0.8, label="Score")
    # annotate cells
    arr = pivot_clean.to_numpy(dtype=float)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if not np.isnan(arr[r, c]):
                ax.text(c, r, f"{arr[r, c]:.3f}", ha="center", va="center", fontsize=8,
                        color="black")
    ax.set_title("Model Performance Summary — All Tasks (Temporal Split)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_5_1_model_metrics_heatmap", "diagnostics")
    reg("fig_5_1_model_metrics_heatmap", "5-Diagnostics",
        "Cross-task model metrics heatmap", paths, inputs=[OCC_RES, RES_TIME_RES])
    plt.show()
    # save LaTeX-ready table
    if not metrics_df.empty:
        pivot.to_csv(TDIR / "model_metrics_latex_ready.csv")
        print("  Saved model_metrics_latex_ready.csv")
except Exception as exc:
    print(f"[ERROR] Fig 5.1: {exc}"); plt.close("all")
""")

# Fig 5.2 — Anomaly scores + map
add_code("""\
try:
    if anomalies.empty or "anomaly_score" not in anomalies.columns:
        raise ValueError("Anomaly scores unavailable")
    scores = pd.to_numeric(anomalies["anomaly_score"], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # histogram
    axes[0].hist(scores, bins=40, color="#DC2626", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Ensemble Anomaly Score")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Anomaly Score Distribution (Isolation Forest)")
    if "anomaly_flag" in anomalies.columns:
        n_anom = anomalies["anomaly_flag"].sum()
        axes[0].axvline(scores[anomalies["anomaly_flag"].astype(bool)].min(),
                        color="black", linestyle="--", linewidth=0.8, label=f"Flag threshold (n={n_anom:,})")
        axes[0].legend()

    # spatial anomaly map
    if not master.empty and col_ok(master, "event_id", "event_lat", "event_lon") and col_ok(anomalies, "event_id"):
        top_anom = anomalies.nlargest(300, "anomaly_score")
        anom_pts = top_anom.merge(master[["event_id", "event_lat", "event_lon"]], on="event_id", how="left").dropna()
        gdf_anom = gpd.GeoDataFrame(
            anom_pts, geometry=gpd.points_from_xy(anom_pts["event_lon"], anom_pts["event_lat"]),
            crs="EPSG:4326"
        )
        gdf_anom = to_proj(gdf_anom)
        axes[1].set_axis_off()
        if not boroughs.empty:
            boroughs.plot(ax=axes[1], color="#F1F5F9", edgecolor="#94A3B8", linewidth=1.0, zorder=1)
        gdf_anom.plot(ax=axes[1], color="#DC2626", markersize=15, alpha=0.6, zorder=2)
        axes[1].set_title("Top 300 Anomalous Events (by Anomaly Score)")
    else:
        axes[1].text(0.5, 0.5, "Spatial anomaly map requires event coordinates",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.suptitle("Anomaly Detection Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_5_2_anomaly_diagnostics", "diagnostics")
    reg("fig_5_2_anomaly_diagnostics", "5-Diagnostics", "Anomaly detection",
        paths, inputs=[ANOMALIES, MASTER])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 5.2: {exc}"); plt.close("all")
""")

# Fig 5.3 — Fairness audit (poverty tercile)
add_code("""\
try:
    if occ_pred.empty or not col_ok(occ_pred, "y_true", "y_pred", "census_poverty_rate"):
        raise ValueError("Occurrence predictions or census_poverty_rate unavailable")
    test_fp = occ_pred[occ_pred["set_name"] == "test"].dropna(
        subset=["y_true", "y_pred", "census_poverty_rate"]).copy()
    if test_fp.empty:
        raise ValueError("No test rows with poverty data")
    test_fp["model_name"] = test_fp["model_name"].astype(str)
    test_fp["poverty_tercile"] = pd.qcut(test_fp["census_poverty_rate"], q=3,
                                          labels=["Low poverty", "Mid poverty", "High poverty"])
    # false negative rate by model × poverty tercile
    def fnr(sub):
        pos = sub[sub["y_true"] == 1]
        if len(pos) == 0: return np.nan
        return (pos["y_pred"] == 0).mean()

    fnr_df = (
        test_fp.groupby(["model_name", "poverty_tercile"])
        .apply(fnr)
        .reset_index(name="FNR")
    )
    pivot_fnr = fnr_df.pivot(index="model_name", columns="poverty_tercile", values="FNR")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (model, row) in enumerate(pivot_fnr.iterrows()):
        x = np.arange(len(row))
        axes[0].plot(x, row.values, marker="o", linewidth=1.5, label=model.replace("_", " "))
    axes[0].set_xticks(range(len(pivot_fnr.columns)))
    axes[0].set_xticklabels(pivot_fnr.columns)
    axes[0].set_ylabel("False Negative Rate"); axes[0].set_xlabel("Poverty Tercile")
    axes[0].set_title("FNR by Poverty Tercile and Model")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # error rate by poverty
    err_df = (
        test_fp.groupby(["model_name", "poverty_tercile"])
        .apply(lambda s: (s["y_true"] != s["y_pred"]).mean())
        .reset_index(name="error_rate")
    )
    pivot_err = err_df.pivot(index="model_name", columns="poverty_tercile", values="error_rate")
    cmap_f = plt.get_cmap("RdYlGn_r", 256)
    im = axes[1].imshow(pivot_err.to_numpy(dtype=float), aspect="auto",
                        cmap=cmap_f, vmin=0, vmax=1)
    axes[1].set_xticks(range(len(pivot_err.columns)))
    axes[1].set_xticklabels(pivot_err.columns, rotation=15)
    axes[1].set_yticks(range(len(pivot_err.index)))
    axes[1].set_yticklabels(pivot_err.index.str.replace("_", " "))
    arr_e = pivot_err.to_numpy(dtype=float)
    for r in range(arr_e.shape[0]):
        for c in range(arr_e.shape[1]):
            if not np.isnan(arr_e[r, c]):
                axes[1].text(c, r, f"{arr_e[r, c]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=axes[1], shrink=0.8, label="Error Rate")
    axes[1].set_title("Error Rate by Poverty Tercile and Model")
    plt.suptitle("Fairness Audit — Occurrence Prediction by Poverty Tercile",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    paths = save_fig(fig, "fig_5_3_fairness_audit_poverty", "diagnostics")
    reg("fig_5_3_fairness_audit_poverty", "5-Diagnostics",
        "Fairness audit by poverty tercile", paths, inputs=[OCC_PRED])
    plt.show()
except Exception as exc:
    print(f"[ERROR] Fig 5.3: {exc}"); plt.close("all")
""")

# ════════════════════════════════════════════════════════════════════════════════
# INVENTORY
# ════════════════════════════════════════════════════════════════════════════════
add_md("## Figure Inventory")

add_code("""\
import datetime
inventory = pd.DataFrame(FIG_REGISTRY)
if not inventory.empty:
    inventory.to_csv(TDIR / "figures_inventory.csv", index=False)
    print(f"Saved figures_inventory.csv — {len(inventory)} figures")
    display(inventory[["figure_id", "section", "title", "created_successfully", "notes"]])
else:
    print("No figures registered.")
print(f"Run completed: {datetime.datetime.now():%Y-%m-%d %H:%M}")
""")

add_md("""\
## Summary

This notebook generates the full visualization suite for the NYC flood infrastructure stress project.

| Section | Figures |
|---------|---------|
| 1 — Exploratory | 8 (study area, network, slope, tide, precip, bubble map, 311 trends, duration) |
| 2 — Clustering | 4 (model selection, PCA, spatial segment map, profiles) |
| 3 — Occurrence + Intensity | 6 (ROC, importance, confusion+calibration, bias, intensity scatter, residuals) |
| 4 — Resolution | 5 (diagnostics, importance, borough MAE, admin violin, spatial map) |
| 5 — Diagnostics | 3 (metrics heatmap, anomaly scores+map, fairness audit) |

All figures saved to `figures/<section>/` as PNG (300 dpi) + PDF.
Figure inventory at `tables/figures_inventory.csv`.
LaTeX-ready metrics table at `tables/model_metrics_latex_ready.csv`.""")

# ── Finalize IDs ──────────────────────────────────────────────────────────────
import hashlib, time
for i, cell in enumerate(cells):
    if cell.get("id") is None or not isinstance(cell.get("id"), str) or len(cell["id"]) < 4:
        cell["id"] = f"cell{i:08d}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "urban-flood-stress",
            "language": "python",
            "name": "urban-flood-stress",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "17_plot.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {out}")
print(f"Cells: {len(cells)}")
