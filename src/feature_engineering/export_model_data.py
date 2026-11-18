
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DIR = ROOT / "data" / "features" / "candidates"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "features" / "model_ready"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for partition in ["train", "student_test"]:
    candidate_folder = CANDIDATE_DIR / partition
    all_candidates = []

    for file in sorted(candidate_folder.glob("*.csv")):
        df = pd.read_csv(file)
        df.insert(0, "trip_id", file.stem)
        df.insert(1, "partition", partition)
        all_candidates.append(df)

    combined = pd.concat(all_candidates, ignore_index=True)

    # Keep candidate rows separate from route sequences.
    combined.to_csv(
        OUTPUT_DIR / f"{partition}_candidates.csv",
        index=False
    )

    print(
        f"{partition}: {len(combined):,} candidate rows, "
        f"{combined['trip_id'].nunique()} trips"
    )

    # Export route sequences separately.
    if partition == "train":
        route_records = []

        for trip_dir in sorted(
            p for p in (RAW_DIR / partition).iterdir() if p.is_dir()
        ):
            trip_id = trip_dir.name

            route = pd.read_csv(
                trip_dir / f"{trip_id}.route",
                sep=r"\s+",
                header=None,
                names=["arc_id"]
            )

            route.insert(0, "route_position", range(len(route)))
            route.insert(0, "trip_id", trip_id)
            route_records.append(route)

        routes = pd.concat(route_records, ignore_index=True)
        routes.to_csv(
            OUTPUT_DIR / "train_route_sequences.csv",
            index=False
        )

print("\nExport completed:", OUTPUT_DIR)

