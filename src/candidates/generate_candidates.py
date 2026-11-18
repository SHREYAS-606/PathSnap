
from pathlib import Path
import pandas as pd
import numpy as np

from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[2]
PREPARED_DIR = ROOT / "data" / "processed" / "prepared"
OUTPUT_DIR = ROOT / "data" / "features" / "candidates"

PARTITIONS = ["train", "student_test"]
TOP_K = 8
RADII = [100, 250]


def generate_trip_candidates(data):
    gps = data["gps"].reset_index(drop=True)
    nodes = data["nodes"]
    arcs = data["arcs"].reset_index()

    # Project coordinates to metres using a local projection
    center_lon = gps["longitude"].mean()
    center_lat = gps["latitude"].mean()

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", local_crs, always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy()
    )

    # Build one directed line geometry per original arc
    geometries = []
    arc_ids = []
    from_nodes = []
    to_nodes = []

    for row in arcs.itertuples(index=False):
        u = int(row.from_node)
        v = int(row.to_node)

        line = LineString([
            (node_x[u], node_y[u]),
            (node_x[v], node_y[v])
        ])

        geometries.append(line)
        arc_ids.append(int(row.arc_id))
        from_nodes.append(u)
        to_nodes.append(v)

    if not geometries:
        raise ValueError("No road arcs found")

    tree = STRtree(geometries)

    # Shapely 1.x and 2.x compatibility
    wkb_to_indices = {}
    for i, geom in enumerate(geometries):
        wkb_to_indices.setdefault(geom.wkb, []).append(i)

    results = []

    gps_x, gps_y = transformer.transform(
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy()
    )

    for obs_id, (x, y) in enumerate(zip(gps_x, gps_y)):
        point = Point(x, y)
        selected = []

        for radius in RADII:
            nearby = tree.query(point.buffer(radius))
            candidate_indices = set()

            for item in nearby:
                if isinstance(item, (int, np.integer)):
                    candidate_indices.add(int(item))
                else:
                    candidate_indices.update(
                        wkb_to_indices.get(item.wkb, [])
                    )

            selected = [
                (i, geometries[i].distance(point))
                for i in candidate_indices
            ]

            selected = [
                item for item in selected if item[1] <= radius
            ]

            if selected:
                break

        selected.sort(key=lambda item: item[1])
        selected = selected[:TOP_K]

        if not selected:
            results.append({
                "observation_id": obs_id,
                "candidate_arc_id": -1,
                "from_node": -1,
                "to_node": -1,
                "distance_m": np.nan,
                "projection_fraction": np.nan,
                "segment_length_m": np.nan
            })
            continue

        for i, distance in selected:
            line = geometries[i]

            results.append({
                "observation_id": obs_id,
                "candidate_arc_id": arc_ids[i],
                "from_node": from_nodes[i],
                "to_node": to_nodes[i],
                "distance_m": distance,
                "projection_fraction": (
                    line.project(point) / line.length
                    if line.length > 0 else 0.0
                ),
                "segment_length_m": line.length
            })

    candidates = pd.DataFrame(results)

    # Attach original GPS information to every candidate
    candidates = candidates.merge(
        gps.reset_index(names="observation_id"),
        on="observation_id",
        how="left"
    )

    return candidates


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_trips = 0
    total_observations = 0
    total_without_candidates = 0

    for partition in PARTITIONS:
        input_dir = PREPARED_DIR / partition
        output_dir = OUTPUT_DIR / partition
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in sorted(input_dir.glob("*.pkl")):
            data = pd.read_pickle(file)

            candidates = generate_trip_candidates(data)

            output_file = output_dir / f"{file.stem}.csv"
            candidates.to_csv(output_file, index=False)

            no_candidate = (
                candidates.loc[
                    candidates["candidate_arc_id"] == -1,
                    "observation_id"
                ].nunique()
            )

            total_trips += 1
            total_observations += len(data["gps"])
            total_without_candidates += no_candidate

            print(
                f"{partition}/{file.stem}: "
                f"{len(data['gps'])} observations, "
                f"{len(candidates)} candidate rows, "
                f"{no_candidate} observations without candidates"
            )

    print("\nCandidate generation complete")
    print("Trips:", total_trips)
    print("GPS observations:", total_observations)
    print("Observations without candidates:", total_without_candidates)


if __name__ == "__main__":
    main()

