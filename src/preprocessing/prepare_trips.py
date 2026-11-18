
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed" / "prepared"

PARTITIONS = ["train", "student_test"]

def load_trip(partition, trip_id):
    trip_dir = RAW_DIR / partition / trip_id

    # Load original files
    gps = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    # Preserve original row order and IDs
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    # Convert values to numeric; fail rather than silently discard rows
    gps = gps.astype(float)
    nodes = nodes.astype(float)
    arcs = arcs.astype(int)
    route = route.astype(int)

    # Validate arc endpoints
    if (
        (arcs["from_node"] < 0).any()
        or (arcs["to_node"] < 0).any()
        or (arcs["from_node"] >= len(nodes)).any()
        or (arcs["to_node"] >= len(nodes)).any()
    ):
        raise ValueError(f"Invalid arc endpoint in {trip_id}")

    # Validate route arc IDs
    if (
        (route["arc_id"] < 0).any()
        or (route["arc_id"] >= len(arcs)).any()
    ):
        raise ValueError(f"Invalid route arc ID in {trip_id}")

    return {
        "trip_id": trip_id,
        "gps": gps,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for partition in PARTITIONS:
        partition_dir = RAW_DIR / partition
        output_partition = OUTPUT_DIR / partition
        output_partition.mkdir(parents=True, exist_ok=True)

        trip_dirs = sorted(
            p for p in partition_dir.iterdir() if p.is_dir()
        )

        for trip_dir in trip_dirs:
            trip_id = trip_dir.name

            data = load_trip(partition, trip_id)

            output_file = output_partition / f"{trip_id}.pkl"
            pd.to_pickle(data, output_file)

            summary.append({
                "partition": partition,
                "trip_id": trip_id,
                "gps_observations": len(data["gps"]),
                "nodes": len(data["nodes"]),
                "arcs": len(data["arcs"]),
                "route_arcs": len(data["route"])
            })

        print(f"{partition}: processed {len(trip_dirs)} trips")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv",
        index=False
    )

    print("\nTotal trips processed:", len(summary_df))
    print("Summary saved:", OUTPUT_DIR / "preprocessing_summary.csv")


if __name__ == "__main__":
    main()
