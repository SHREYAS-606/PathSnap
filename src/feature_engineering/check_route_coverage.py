
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "train"
CANDIDATE_DIR = ROOT / "data" / "features" / "candidates" / "train"

results = []

for trip_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
    trip_id = trip_dir.name

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    candidates = pd.read_csv(
        CANDIDATE_DIR / f"{trip_id}.csv"
    )

    route_arc_ids = set(route["arc_id"].astype(int))

    valid = candidates[candidates["candidate_arc_id"] >= 0].copy()

    # Diagnostic only: route arc appears somewhere in this trip's route
    valid["route_arc_present"] = (
        valid["candidate_arc_id"].isin(route_arc_ids)
    )

    observation_coverage = (
        valid.groupby("observation_id")["route_arc_present"].any()
    )

    results.append({
        "trip_id": trip_id,
        "observations": candidates["observation_id"].nunique(),
        "observations_with_route_candidate": int(observation_coverage.sum()),
        "coverage_percent": round(
            100 * observation_coverage.mean(), 2
        )
    })

result = pd.DataFrame(results)

output = ROOT / "data" / "processed" / "route_candidate_coverage.csv"
result.to_csv(output, index=False)

print(result.to_string(index=False))
print("\nOverall coverage:",
      round(
          100 * result["observations_with_route_candidate"].sum()
          / result["observations"].sum(), 2
      ), "%")
print("Saved:", output)

