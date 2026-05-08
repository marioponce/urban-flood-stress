from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.flood_event_balancing import (  # noqa: E402
    BALANCED_GEOPARQUET_PATH,
    BALANCED_PARQUET_PATH,
    UNREPORTED_GEOPARQUET_PATH,
    UNREPORTED_PARQUET_PATH,
    VALIDATION_BALANCING_PATH,
    VALIDATION_MASTER_PATH,
    VALIDATION_UNREPORTED_PATH,
    build_balanced_dataset,
)


def main() -> None:
    balanced, unreported, validation_balancing, _ = build_balanced_dataset(write_outputs=True)
    print(f"balanced_rows={len(balanced)}")
    print(f"negative_rows={int((balanced['occurrence'] == False).sum())}")  # noqa: E712
    print(f"positive_rows={int((balanced['occurrence'] == True).sum())}")  # noqa: E712
    print(f"unreported_rows={len(unreported)}")
    print(f"balanced_parquet={BALANCED_PARQUET_PATH}")
    print(f"balanced_geoparquet={BALANCED_GEOPARQUET_PATH}")
    print(f"unreported_parquet={UNREPORTED_PARQUET_PATH}")
    print(f"unreported_geoparquet={UNREPORTED_GEOPARQUET_PATH}")
    print(f"validation_master={VALIDATION_MASTER_PATH}")
    print(f"validation_balancing={VALIDATION_BALANCING_PATH}")
    print(f"validation_unreported={VALIDATION_UNREPORTED_PATH}")
    print(validation_balancing.to_string(index=False))


if __name__ == "__main__":
    main()
