from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/street-flooding.ipynb")


def cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        text = cell_source(cell)
        if "def prepare_individual_complaints()" not in text:
            continue

        old = """def prepare_individual_complaints() -> tuple[pd.DataFrame, dict[str, int], dict[str, str | None], str]:
    \"\"\"Build the complaint-level table with spatial and textual matching.\"\"\"
    cached = pd.read_csv(INDIVIDUAL_PATH, low_memory=False) if INDIVIDUAL_PATH.exists() else None
    if cached is not None and {'complaint_id', 'segment_id', 'start', 'status'}.issubset(cached.columns):
        cached['start'] = pd.to_datetime(cached['start'], errors='coerce')
        cached['end'] = pd.to_datetime(cached['end'], errors='coerce')
        cached['segment_id'] = cached['segment_id'].astype('string')
        cached['match_method'] = cached['match_method'].astype('string') if 'match_method' in cached.columns else pd.Series(pd.NA, index=cached.index, dtype='string')
        cached['match_distance'] = pd.to_numeric(cached['match_distance'], errors='coerce') if 'match_distance' in cached.columns else np.nan
        cached['status'] = pd.to_numeric(cached['status'], errors='coerce').fillna(0).astype('int8')
        return cached, {}, {}, 'cached'
"""

        new = """def prepare_individual_complaints() -> tuple[pd.DataFrame, dict[str, int], dict[str, str | None], str]:
    \"\"\"Build the complaint-level table with spatial and textual matching.\"\"\"
    cached = pd.read_csv(INDIVIDUAL_PATH, low_memory=False) if INDIVIDUAL_PATH.exists() else None
    if cached is not None and {'complaint_id', 'segment_id', 'start', 'status'}.issubset(cached.columns):
        cached = cached.copy()
        if 'end' not in cached.columns:
            cached['end'] = pd.NaT
        source_id = 'cached'
        if 'source_id' in cached.columns and cached['source_id'].notna().any():
            source_id = cached['source_id'].dropna().astype('string').iloc[0]
        elif cached.attrs.get('source_id'):
            source_id = str(cached.attrs['source_id'])
        field_map = infer_field_map(cached.columns, source_id)
        cached['start'] = pd.to_datetime(cached['start'], errors='coerce')
        cached['end'] = pd.to_datetime(cached['end'], errors='coerce')
        cached['segment_id'] = cached['segment_id'].astype('string')
        cached['match_method'] = cached['match_method'].astype('string') if 'match_method' in cached.columns else pd.Series(pd.NA, index=cached.index, dtype='string')
        cached['match_distance'] = pd.to_numeric(cached['match_distance'], errors='coerce') if 'match_distance' in cached.columns else pd.Series(np.nan, index=cached.index)
        cached['status'] = pd.to_numeric(cached['status'], errors='coerce').fillna(0).astype('int8')
        for column in [
            'normalized_street_name',
            'borough_normalized',
            'borough',
            'street_name_raw',
            'status_text',
            'coordinate_source',
            'has_valid_coordinates',
            'latitude',
            'longitude',
            'x_coordinate_state_plane',
            'y_coordinate_state_plane',
            'source_id',
            'source_window',
        ]:
            if column not in cached.columns:
                cached[column] = pd.NA
        cached['source_id'] = cached['source_id'].fillna(source_id)
        return cached, {}, field_map, source_id
"""

        if old not in text:
            raise RuntimeError("Could not find the cached complaint preparation block to patch.")

        text = text.replace(old, new)
        cell["source"] = [f"{line}\n" for line in text.rstrip("\n").splitlines()]
        cell["execution_count"] = None
        cell["outputs"] = []
        break
    else:
        raise RuntimeError("Could not find prepare_individual_complaints in the notebook.")

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
