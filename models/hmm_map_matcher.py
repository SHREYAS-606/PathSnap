
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


ROOT = Path(__file__).resolve().parents[1]

PREPARED_DIR = ROOT / "data" / "processed" / "prepared"
CANDIDATE_DIR = ROOT / "data" / "features" / "candidates"
OUTPUT_DIR = ROOT / "outputs" / "hmm"


# HMM hyperparameters
EMISSION_SIGMA = 15.0
TRANSITION_BETA = 50.0
TRANSITION_WEIGHT = 0.2
UNREACHABLE_PENALTY = -20.0


def haversine(lon1, lat1, lon2, lat2):
    """Calculate geographical distance in metres."""

    radius = 6371000.0

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

    return 2 * radius * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def build_road_graph(arcs, nodes):
    """Build a directed, distance-weighted road graph."""

    node_ids = nodes.index.to_numpy(dtype=int)
    node_count = int(node_ids.max()) + 1

    node_lon = nodes["longitude"].to_dict()
    node_lat = nodes["latitude"].to_dict()

    edges = arcs.reset_index().copy()

    if "arc_id" not in edges.columns:
        raise ValueError("Road arcs must contain arc_id.")

    edges["from_node"] = edges["from_node"].astype(int)
    edges["to_node"] = edges["to_node"].astype(int)

    valid = (
        edges["from_node"].isin(node_ids)
        & edges["to_node"].isin(node_ids)
    )

    edges = edges[valid].copy()

    edges["length_m"] = haversine(
        edges["from_node"].map(node_lon).to_numpy(),
        edges["from_node"].map(node_lat).to_numpy(),
        edges["to_node"].map(node_lon).to_numpy(),
        edges["to_node"].map(node_lat).to_numpy()
    )

    # Keep the shortest edge for duplicate directed pairs.
    edges = edges.groupby(
        ["from_node", "to_node"], as_index=False
    )["length_m"].min()

    graph = csr_matrix(
        (
            edges["length_m"].to_numpy(),
            (
                edges["from_node"].to_numpy(),
                edges["to_node"].to_numpy()
            )
        ),
        shape=(node_count, node_count)
    )

    return graph


def emission_log_probability(distance):
    """Gaussian GPS-to-road emission log probability."""

    distance = np.asarray(distance, dtype=float)

    return (
        -0.5 * (distance / EMISSION_SIGMA) ** 2
        - math.log(EMISSION_SIGMA * math.sqrt(2 * math.pi))
    )


def transition_log_probability(
    previous,
    current,
    gps_distance,
    graph,
    shortest_paths
):
    """Calculate transition score using directed shortest paths."""

    previous_arc = int(previous["candidate_arc_id"])
    current_arc = int(current["candidate_arc_id"])

    previous_fraction = float(
        previous["projection_fraction"]
    )

    current_fraction = float(
        current["projection_fraction"]
    )

    previous_length = float(
        previous["segment_length_m"]
    )

    current_length = float(
        current["segment_length_m"]
    )

    # Same directed arc, moving forward.
    if (
        previous_arc == current_arc
        and current_fraction >= previous_fraction
    ):
        network_distance = (
            current_fraction - previous_fraction
        ) * previous_length

    else:
        previous_to = int(previous["to_node"])
        current_from = int(current["from_node"])

        if previous_to == current_from:
            middle_distance = 0.0

        else:
            if previous_to not in shortest_paths:
                return UNREACHABLE_PENALTY

            middle_distance = shortest_paths[
    previous_to
][current_from]

            if not np.isfinite(middle_distance):
                return UNREACHABLE_PENALTY

        remaining_previous = (
            (1.0 - previous_fraction)
            * previous_length
        )

        distance_on_current = (
            current_fraction * current_length
        )

        network_distance = (
            remaining_previous
            + middle_distance
            + distance_on_current
        )

    beta = max(
        TRANSITION_BETA,
        gps_distance * 0.5 + 20.0
    )

    return (
        -abs(network_distance - gps_distance) / beta
    )


def prepare_observations(candidates):
    """Clean candidates and group them by GPS observation."""

    required = [
        "observation_id",
        "candidate_arc_id",
        "from_node",
        "to_node",
        "distance_m",
        "projection_fraction",
        "segment_length_m",
        "longitude",
        "latitude",
        "timestamp"
    ]

    missing = [
        column for column in required
        if column not in candidates.columns
    ]

    if missing:
        raise ValueError(
            f"Missing candidate columns: {missing}"
        )

    candidates = candidates.copy()

    for column in required:
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce"
        )

    candidates = candidates.dropna(
        subset=required
    )

    candidates = candidates[
        (candidates["candidate_arc_id"] >= 0)
        & (candidates["from_node"] >= 0)
        & (candidates["to_node"] >= 0)
        & (candidates["distance_m"] >= 0)
        & (candidates["segment_length_m"] > 0)
    ].copy()

    candidates = candidates.sort_values(
        ["observation_id", "distance_m"]
    )

    groups = []

    for observation_id, group in candidates.groupby(
        "observation_id",
        sort=True
    ):
        group = group.drop_duplicates(
            subset=["candidate_arc_id"]
        ).reset_index(drop=True)

        if not group.empty:
            groups.append({
                "observation_id": int(observation_id),
                "candidates": group
            })

    return groups


def viterbi_decode(groups, graph):
    """Find the most likely road-segment sequence."""

    if not groups:
        return []

    # Initialize scores using emission probabilities.
    first = groups[0]["candidates"]

    scores = emission_log_probability(
        first["distance_m"].to_numpy()
    )

    backpointers = []

    for index in range(1, len(groups)):

        previous_group = groups[index - 1]["candidates"]
        current_group = groups[index]["candidates"]

        current_obs = groups[index]["observation_id"]

        previous_gps = previous_group.iloc[0]
        current_gps = current_group.iloc[0]

        gps_distance = haversine(
            previous_gps["longitude"],
            previous_gps["latitude"],
            current_gps["longitude"],
            current_gps["latitude"]
        )

        # Cache shortest paths for each previous node
        # during this observation transition.
        shortest_paths = {}

        transition_scores = np.full(
            (
                len(previous_group),
                len(current_group)
            ),
            UNREACHABLE_PENALTY,
            dtype=float
        )

        for i, previous in previous_group.iterrows():

            previous_to = int(previous["to_node"])

            if previous_to not in shortest_paths:

                maximum_distance = max(
                    gps_distance * 4.0 + 1000.0,
                    1500.0
                )

                shortest_paths[previous_to] = dijkstra(
                    graph,
                    directed=True,
                    indices=previous_to,
                    limit=maximum_distance
                )

            paths = shortest_paths[previous_to]

            for j, current in current_group.iterrows():

                transition = transition_log_probability(
                    previous,
                    current,
                    gps_distance,
                    graph,
                    {previous_to: paths}
                )

                # Preserve a strong penalty for unreachable
                # transitions. Weight only reachable scores.
                if transition == UNREACHABLE_PENALTY:
                    transition_scores[i, j] = (
                        UNREACHABLE_PENALTY
                    )
                else:
                    transition_scores[i, j] = (
                        TRANSITION_WEIGHT * transition
                    )

        # Emission probabilities for current observation.
        emission_scores = emission_log_probability(
            current_group["distance_m"].to_numpy()
        )

        # Viterbi forward recurrence.
        combined = (
            scores[:, None] + transition_scores
        )

        best_previous = np.argmax(
            combined,
            axis=0
        )

        scores = (
            combined[
                best_previous,
                np.arange(len(current_group))
            ]
            + emission_scores
        )

        backpointers.append(best_previous)

        print(
            f"Viterbi: observation {current_obs} "
            f"processed ({len(current_group)} states)"
        )

    # Backtracking.
    state = int(np.argmax(scores))

    selected_indices = [state]

    for pointers in reversed(backpointers):
        state = int(pointers[state])
        selected_indices.append(state)

    selected_indices.reverse()

    results = []

    for group, state_index in zip(
        groups,
        selected_indices
    ):

        selected = group["candidates"].iloc[
            state_index
        ]

        results.append({
            "observation_id": group["observation_id"],
            "predicted_arc_id": int(
                selected["candidate_arc_id"]
            ),
            "from_node": int(
                selected["from_node"]
            ),
            "to_node": int(
                selected["to_node"]
            ),
            "distance_m": float(
                selected["distance_m"]
            ),
            "projection_fraction": float(
                selected["projection_fraction"]
            )
        })

    return results


def match_trip(trip_id, partition):
    """Run HMM route reconstruction for one trip."""

    trip_name = str(trip_id).zfill(8)

    prepared_file = (
        PREPARED_DIR / partition / f"{trip_name}.pkl"
    )

    candidate_file = (
        CANDIDATE_DIR / partition / f"{trip_name}.csv"
    )

    if not prepared_file.exists():
        raise FileNotFoundError(prepared_file)

    if not candidate_file.exists():
        raise FileNotFoundError(candidate_file)

    print(f"\nLoading trip {trip_name}...")

    data = pd.read_pickle(prepared_file)
    candidates = pd.read_csv(candidate_file)

    print("GPS observations:", len(data["gps"]))
    print("Candidate rows:", len(candidates))

    graph = build_road_graph(
        data["arcs"],
        data["nodes"]
    )

    print(
        "Road graph:",
        graph.shape,
        "nodes;",
        graph.nnz,
        "directed connections"
    )

    groups = prepare_observations(candidates)

    print("Usable GPS observations:", len(groups))

    predictions = viterbi_decode(
        groups,
        graph
    )

    result = pd.DataFrame(predictions)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR / f"{partition}_{trip_name}_hmm.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print("\nHMM route reconstruction complete")
    print("Predicted observations:", len(result))
    print("Output:", output_file)

    return result


def main():

    parser = argparse.ArgumentParser(
        description="PathSnap HMM and Viterbi Map Matcher"
    )

    parser.add_argument(
        "--trip-id",
        default="00000001"
    )

    parser.add_argument(
        "--partition",
        choices=["train", "student_test"],
        default="train"
    )

    args = parser.parse_args()

    match_trip(
        args.trip_id,
        args.partition
    )


if __name__ == "__main__":
    main()
