import pandas as pd
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("igraph")
pytest.importorskip("shapely")

import geopandas as gpd
from shapely.geometry import LineString

from project_name.street_network import (
    compute_directed_graph_metrics,
    download_lion_street_network,
    prepare_lion_street_network,
)


def _street_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "FeatureTyp": "0",
                "SegmentTyp": "R",
                "RW_TYPE": "12",
                "TrafDir": "T",
                "NodeLevelF": "1",
                "NodeLevelT": "1",
                "POSTED_SPEED": "30",
                "Number_Total_Lanes": "4",
                "geometry": LineString([(0, 0), (10, 0)]),
            },
            {
                "FeatureTyp": "0",
                "SegmentTyp": "R",
                "RW_TYPE": "5",
                "TrafDir": "W",
                "NodeLevelF": "1",
                "NodeLevelT": "1",
                "POSTED_SPEED": "20",
                "Number_Total_Lanes": "2",
                "geometry": LineString([(10, 0), (20, 0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:2263",
    )


def test_prepare_lion_street_network_builds_graph_columns():
    prepared = prepare_lion_street_network(_street_frame())

    assert list(prepared["edge_id"]) == [0, 1]
    assert prepared["road_class"].tolist() == ["highway", "collector"]
    assert prepared["trafdir_resolved"].tolist() == ["T", "W"]
    assert prepared["in_graph"].tolist() == [True, True]
    assert pd.api.types.is_integer_dtype(prepared["u"].dtype)
    assert pd.api.types.is_integer_dtype(prepared["v"].dtype)
    assert prepared.loc[0, "u"] == 0
    assert prepared.loc[0, "v"] == 1
    assert prepared.loc[1, "u"] == 1
    assert prepared.loc[1, "v"] == 2


def test_compute_directed_graph_metrics_returns_street_and_node_tables():
    streets, node_metrics, edge_metrics = compute_directed_graph_metrics(
        _street_frame()
    )

    assert "edge_betweenness" in streets.columns
    assert streets["edge_betweenness"].notna().all()
    assert len(node_metrics) == 3
    assert set(node_metrics.columns) == {
        "node_id",
        "in_degree",
        "out_degree",
        "total_degree",
        "node_betweenness",
    }
    assert len(edge_metrics) == 2
    assert set(edge_metrics.columns) == {"edge_id", "edge_betweenness"}


def test_download_lion_street_network_writes_raw_and_geodata(
    monkeypatch, tmp_path
):
    raw_rows = pd.DataFrame(
        [
            {
                "FeatureTyp": "0",
                "SegmentTyp": "R",
                "RW_TYPE": "12",
                "TrafDir": "T",
                "NodeLevelF": "1",
                "NodeLevelT": "1",
                "POSTED_SPEED": "30",
                "Number_Total_Lanes": "4",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [10, 0]],
                },
            }
        ]
    )

    written = {}

    def fake_fetch(*args, **kwargs):
        return raw_rows

    def fake_to_csv(self, path, index=False):
        written["csv_path"] = str(path)
        written["csv_index"] = index

    def fake_to_file(self, path, driver=None):
        written["vector_path"] = str(path)
        written["vector_driver"] = driver

    monkeypatch.setattr("project_name.street_network.fetch_socrata_rows", fake_fetch)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

    raw_frame, geodata = download_lion_street_network(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        raw_name="lion",
        target_crs=None,
    )

    assert not raw_frame.empty
    assert geodata is not None
    assert written["csv_path"].endswith("lion.csv")
    assert written["csv_index"] is False
    assert written["vector_path"].endswith("lion.gpkg")
    assert written["vector_driver"] == "GPKG"
