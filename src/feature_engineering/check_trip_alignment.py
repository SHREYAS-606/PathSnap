
from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train"

for trip_id in ["00000054", "00000033"]:
    folder = RAW / trip_id

    gps = pd.read_csv(
        folder / f"{trip_id}.track",
        sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )
    nodes = pd.read_csv(
        folder / f"{trip_id}.nodes",
        sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )
    arcs = pd.read_csv(
        folder / f"{trip_id}.arcs",
        sep=r"\s+", header=None,
        names=["from_node", "to_node"]
    )
    route = pd.read_csv(
        folder / f"{trip_id}.route",
        sep=r"\s+", header=None, names=["arc_id"]
    )

    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={gps.latitude.mean()} "
        f"+lon_0={gps.longitude.mean()} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )

    nx, ny = transformer.transform(
        nodes.longitude.to_numpy(),
        nodes.latitude.to_numpy()
    )

    route_points = []
    for arc_id in route.arc_id.astype(int):
        u, v = arcs.iloc[arc_id]
        route_points.append((nx[int(u)], ny[int(u)]))

    last_arc = int(route.arc_id.iloc[-1])
    last_node = int(arcs.iloc[last_arc]["to_node"])
    route_points.append((nx[last_node], ny[last_node]))

    route_line = LineString(route_points)

    gx, gy = transformer.transform(
        gps.longitude.to_numpy(),
        gps.latitude.to_numpy()
    )

    distances = [
        route_line.distance(Point(x, y))
        for x, y in zip(gx, gy)
    ]

    print(f"\nTRIP {trip_id}")
    print("GPS points:", len(gps))
    print("Route arcs:", len(route))
    print("GPS duration (seconds):",
          round(gps.timestamp.max() - gps.timestamp.min(), 2))
    print("Median distance to route (m):",
          round(float(np.median(distances)), 2))
    print("95th percentile distance (m):",
          round(float(np.percentile(distances, 95)), 2))


