
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PREPARED_DIR = ROOT / "data" / "processed" / "prepared"
CANDIDATE_DIR = ROOT / "data" / "features" / "candidates"
OUTPUT_DIR = ROOT / "data" / "features" / "engineered"

PARTITIONS = ["train", "student_test"]


def bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def haversine(lon1, lat1, lon2, lat2):
    radius = 6371000

    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def engineer_trip(data, candidates):
    gps = data["gps"].reset_index(drop=True)
    nodes = data["nodes"]
    arcs = data["arcs"].reset_index(drop=True)

    # Calculate GPS movement features
    previous_lon = gps["longitude"].shift()
    previous_lat = gps["latitude"].shift()
    previous_time = gps["timestamp"].shift()

    candidates["time_gap_s"] = (
        gps["timestamp"] - previous_time
    ).reindex(candidates["observation_id"]).to_numpy()

    movement_distance = haversine(
        previous_lon.to_numpy(),
        previous_lat.to_numpy(),
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy()
    )

    gps_speed = movement_distance / (
        gps["timestamp"] - previous_time
    ).replace(0, np.nan).to_numpy()

    gps_heading = bearing(
        previous_lon.to_numpy(),
        previous_lat.to_numpy(),
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy()
    )

    # Ignore heading and speed when the previous GPS point
    # is unavailable or the time interval is invalid.
    valid_time = (
        gps["timestamp"].diff().to_numpy() > 0
    )

    gps_speed[~valid_time] = np.nan
    gps_heading[~valid_time] = np.nan

    candidates["gps_speed_mps"] = (
        candidates["observation_id"].map(
            dict(enumerate(gps_speed))
        )
    )

    candidates["gps_heading_deg"] = (
        candidates["observation_id"].map(
            dict(enumerate(gps_heading))
        )
    )

    # Directed road connectivity
    from_nodes = arcs["from_node"].to_numpy(dtype=int)
    to_nodes = arcs["to_node"].to_numpy(dtype=int)

    node_count = len(nodes)

    out_degree = np.bincount(
        from_nodes, minlength=node_count
    )
    in_degree = np.bincount(
        to_nodes, minlength=node_count
    )

    # Road heading from start node to end node
    node_lon = nodes["longitude"].to_numpy()
    node_lat = nodes["latitude"].to_numpy()

    road_heading = bearing(
        node_lon[from_nodes],
        node_lat[from_nodes],
        node_lon[to_nodes],
        node_lat[to_nodes]
    )

    road_length = haversine(
        node_lon[from_nodes],
        node_lat[from_nodes],
        node_lon[to_nodes],
        node_lat[to_nodes]
    )

    arc_ids = candidates["candidate_arc_id"].to_numpy(dtype=int)
    valid_arc = (arc_ids >= 0) & (arc_ids < len(arcs))

    candidates["road_heading_deg"] = np.nan
    candidates["road_length_m"] = np.nan
    candidates["from_node_out_degree"] = np.nan
    candidates["to_node_in_degree"] = np.nan

    candidates.loc[valid_arc, "road_heading_deg"] = (
        road_heading[arc_ids[valid_arc]]
    )
    candidates.loc[valid_arc, "road_length_m"] = (
        road_length[arc_ids[valid_arc]]
    )
    candidates.loc[valid_arc, "from_node_out_degree"] = (
        out_degree[from_nodes[arc_ids[valid_arc]]]
    )
    candidates.loc[valid_arc, "to_node_in_degree"] = (
        in_degree[to_nodes[arc_ids[valid_arc]]]
    )

    # Circular difference between GPS and road directions
    difference = (
        candidates["gps_heading_deg"]
        - candidates["road_heading_deg"]
    ).abs()

    candidates["heading_difference_deg"] = np.minimum(
        difference, 360 - difference
    )

    # Useful distance transformations
    candidates["distance_squared"] = (
        candidates["distance_m"] ** 2
    )

    candidates["is_valid_candidate"] = (
        candidates["candidate_arc_id"] >= 0
    ).astype(int)

    # Flag unusually long GPS sampling intervals
    candidates["large_time_gap"] = (
        candidates["time_gap_s"] > 5
    ).astype(int)

    return candidates


def main():
    for partition in PARTITIONS:
        input_dir = CANDIDATE_DIR / partition
        output_dir = OUTPUT_DIR / partition
        output_dir.mkdir(parents=True, exist_ok=True)

        all_features = []

        for file in sorted(input_dir.glob("*.csv")):
            trip_id = file.stem

            data = pd.read_pickle(
                PREPARED_DIR / partition / f"{trip_id}.pkl"
            )

            candidates = pd.read_csv(file)

            features = engineer_trip(data, candidates)

            features.insert(0, "trip_id", trip_id)
            features.insert(1, "partition", partition)

            features.to_csv(
                output_dir / f"{trip_id}.csv",
                index=False
            )

            all_features.append(features)

            print(
                f"{partition}/{trip_id}: "
                f"{len(features)} feature rows"
            )

        combined = pd.concat(all_features, ignore_index=True)

        combined.to_csv(
            OUTPUT_DIR / f"{partition}_features.csv",
            index=False
        )

        print(
            f"\n{partition}: exported "
            f"{len(combined):,} rows\n"
        )


if __name__ == "__main__":
    main()
