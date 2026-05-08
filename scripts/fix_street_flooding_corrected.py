from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/street-flooding_corrected.ipynb")


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
        if "def merge_segment_intervals" not in text or "def prepare_flood_events" not in text:
            continue

        text = replace_once(
            text,
            """def merge_segment_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Merge overlapping complaint intervals for one segment.\"\"\"
    output_columns = ["segment_id", "start", "end", "n_complaints", "status"]

    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    group = frame.sort_values(["start", "end"], kind="stable").reset_index(drop=True).copy()

    start_ns = group["start"].astype("int64").to_numpy()
    end_ns = group["end"].astype("int64").to_numpy()
    open_ended = group["end"].isna().to_numpy()
    end_for_merging = np.where(open_ended, np.iinfo(np.int64).max, end_ns)

    running_max = np.maximum.accumulate(end_for_merging)

    previous_max = np.empty_like(running_max)
    previous_max[0] = np.iinfo(np.int64).min
    previous_max[1:] = running_max[:-1]

    new_event = start_ns > previous_max
    group["local_event_id"] = new_event.cumsum() - 1

    aggregated = (
        group.groupby("local_event_id", sort=True)
        .agg(
            segment_id=("segment_id", "first"),
            start=("start", "min"),
            end=("end", lambda s: pd.NaT if s.isna().any() else s.max()),
            n_complaints=("segment_id", "size"),
            n_open_complaints=("status", lambda s: int((s == 0).sum())),
            n_closed_complaints=("status", lambda s: int((s == 1).sum())),
        )
        .reset_index(drop=True)
    )

    aggregated["status"] = np.select(
        [
            aggregated["n_closed_complaints"].eq(0),
            aggregated["n_open_complaints"].eq(0),
        ],
        [
            0,
            1,
        ],
        default=2,
    ).astype("int8")

    return aggregated[output_columns]


def prepare_flood_events(individual: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complaint intervals into non-overlapping flood events by segment_id."""
    usable = individual[individual["segment_id"].notna()].copy()

    # Required ordering.
    usable = usable.sort_values(["segment_id", "start"], kind="stable").reset_index(drop=True)

    event_parts = []

    for _, group in usable.groupby("segment_id", sort=False):
        merged = merge_segment_intervals(group)

        if not merged.empty:
            event_parts.append(merged)

    if event_parts:
        events = pd.concat(event_parts, ignore_index=True)
    else:
        events = pd.DataFrame(columns=["segment_id", "start", "end", "n_complaints", "status"])

    events = events.sort_values(["segment_id", "start"], kind="stable").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype="int64")

    events = events[
        [
            "event_id",
            "start",
            "end",
            "segment_id",
            "n_complaints",
            "status",
        ]
    ].copy()

    events.to_csv(EVENTS_PATH, index=False)

    return events
""",
            """def merge_segment_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Merge complaint intervals for one segment and close open-only events by observed start span.\"\"\"
    output_columns = ["segment_id", "start", "end", "duration_hours", "n_complaints", "status"]

    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    group = frame.copy()
    group["start"] = pd.to_datetime(group["start"], utc=True, errors="coerce")
    group["end"] = pd.to_datetime(group["end"], utc=True, errors="coerce")
    group = group.dropna(subset=["start"]).sort_values(["start", "end"], kind="stable").reset_index(drop=True)

    if group.empty:
        return pd.DataFrame(columns=output_columns)

    if group["end"].notna().any():
        start_ns = group["start"].astype("int64").to_numpy()
        end_ns = group["end"].astype("int64").to_numpy()
        open_ended = group["end"].isna().to_numpy()
        end_for_merging = np.where(open_ended, np.iinfo(np.int64).max, end_ns)

        running_max = np.maximum.accumulate(end_for_merging)
        previous_max = np.empty_like(running_max)
        previous_max[0] = np.iinfo(np.int64).min
        previous_max[1:] = running_max[:-1]

        new_event = start_ns > previous_max
        group["local_event_id"] = new_event.cumsum() - 1

        aggregated = (
            group.groupby("local_event_id", sort=True)
            .agg(
                segment_id=("segment_id", "first"),
                start=("start", "min"),
                end=("end", lambda s: pd.NaT if s.isna().any() else s.max()),
                n_complaints=("segment_id", "size"),
                n_open_complaints=("status", lambda s: int((s == 0).sum())),
                n_closed_complaints=("status", lambda s: int((s == 1).sum())),
            )
            .reset_index(drop=True)
        )
    else:
        # The source feed does not expose closed timestamps, so close events by
        # the observed span of complaint starts on each segment with a 24h gap.
        gap = pd.Timedelta(hours=24)
        start_diff = group["start"].diff()
        new_event = start_diff.isna() | (start_diff > gap)
        group["local_event_id"] = new_event.cumsum() - 1

        aggregated = (
            group.groupby("local_event_id", sort=True)
            .agg(
                segment_id=("segment_id", "first"),
                start=("start", "min"),
                end=("start", "max"),
                n_complaints=("segment_id", "size"),
                n_open_complaints=("status", lambda s: int((s == 0).sum())),
                n_closed_complaints=("status", lambda s: int((s == 1).sum())),
            )
            .reset_index(drop=True)
        )

    aggregated["duration_hours"] = (
        (aggregated["end"] - aggregated["start"]).dt.total_seconds() / 3600.0
    )
    aggregated.loc[aggregated["end"].isna(), "duration_hours"] = pd.NA

    aggregated["status"] = np.select(
        [
            aggregated["n_closed_complaints"].eq(0),
            aggregated["n_open_complaints"].eq(0),
        ],
        [
            0,
            1,
        ],
        default=2,
    ).astype("int8")

    return aggregated[output_columns]


def prepare_flood_events(individual: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complaint intervals into non-overlapping flood events by segment_id."""
    usable = individual[individual["segment_id"].notna()].copy()

    # Required ordering.
    usable = usable.sort_values(["segment_id", "start"], kind="stable").reset_index(drop=True)

    event_parts = []

    for _, group in usable.groupby("segment_id", sort=False):
        merged = merge_segment_intervals(group)

        if not merged.empty:
            event_parts.append(merged)

    if event_parts:
        events = pd.concat(event_parts, ignore_index=True)
    else:
        events = pd.DataFrame(columns=["segment_id", "start", "end", "duration_hours", "n_complaints", "status"])

    events = events.sort_values(["segment_id", "start"], kind="stable").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype="int64")

    events = events[
        [
            "event_id",
            "start",
            "end",
            "duration_hours",
            "segment_id",
            "n_complaints",
            "status",
        ]
    ].copy()

    events.to_csv(EVENTS_PATH, index=False)

    return events
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
