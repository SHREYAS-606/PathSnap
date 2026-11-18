
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw" / "train"
CANDIDATES = ROOT / "data" / "features" / "candidates" / "train"

for trip_id in ["00000054", "00000033", "00000072", "00000002"]:
    trip_dir = RAW / trip_id

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+", header=None, names=["arc_id"]
    )

    candidates = pd.read_csv(
        CANDIDATES / f"{trip_id}.csv"
    )

    route_ids = set(route["arc_id"].astype(int))
    valid = candidates[candidates["candidate_arc_id"] >= 0].copy()

    valid["is_route_arc"] = valid["candidate_arc_id"].isin(route_ids)

    per_observation = valid.groupby("observation_id").agg(
        candidate_count=("candidate_arc_id", "count"),
        route_candidates=("is_route_arc", "sum"),
        nearest_distance_m=("distance_m", "min")
    )

    print(f"\n{'=' * 50}")
    print("TRIP:", trip_id)
    print("Route arcs:", len(route))
    print("Unique route arcs:", len(route_ids))
    print("GPS observations:", len(per_observation))
    print("Observations with route candidates:",
          (per_observation["route_candidates"] > 0).sum())
    print("Median nearest candidate distance:",
          round(per_observation["nearest_distance_m"].median(), 2))
    print("95th percentile nearest distance:",
          round(per_observation["nearest_distance_m"].quantile(0.95), 2))

