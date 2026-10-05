"""Vectorised Mobike ride and parking-location analysis.

The default paths and the four legacy CSV filenames are kept for compatibility
with the original script.  Run ``python mobikeAnalyzier_optimized.py --help``
for all adjustable clustering and path parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

# Use a non-interactive backend so the script also works on servers and CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


EARTH_RADIUS_KM = 6371.0088
SHANGHAI_CENTER = (121.471632, 31.233705)  # longitude, latitude
REQUIRED_COLUMNS = {
    "orderid",
    "bikeid",
    "userid",
    "start_time",
    "end_time",
    "start_location_x",
    "start_location_y",
    "end_location_x",
    "end_location_y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse Mobike ride records.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mobike_shanghai_sample_updated.csv"),
        help="Input CSV path (default: data/mobike_shanghai_sample_updated.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for CSV and PNG outputs (default: data).",
    )
    parser.add_argument("--knn-neighbors", type=int, default=200)
    parser.add_argument("--kmeans-clusters", type=int, default=70)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--dbscan-eps-km", type=float, default=0.30)
    parser.add_argument("--dbscan-min-samples", type=int, default=200)
    parser.add_argument(
        "--border-radius-km",
        type=float,
        default=0.30,
        help=(
            "Distance from a KMeans centre used to select border points. "
            "0.30 km is the geographic approximation of original 0.003 degrees."
        ),
    )
    parser.add_argument("--border-tolerance-km", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def haversine_km(
    lon1: np.ndarray | pd.Series | float,
    lat1: np.ndarray | pd.Series | float,
    lon2: np.ndarray | pd.Series | float,
    lat2: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Return great-circle distance in kilometres; accepts NumPy-broadcastable inputs."""
    # Convert independently: a Series paired with a scalar is a valid broadcast
    # case, while passing such a mixed list to np.deg2rad is not.
    lon1_rad = np.deg2rad(lon1)
    lat1_rad = np.deg2rad(lat1)
    lon2_rad = np.deg2rad(lon2)
    lat2_rad = np.deg2rad(lat2)
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def validate_and_load(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path.resolve()}")
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    for column in ("orderid", "bikeid", "userid"):
        # Pandas StringDtype preserves missing values instead of turning them into 'nan'.
        data[column] = data[column].astype("string")
    for column in ("start_time", "end_time"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    for column in (
        "start_location_x",
        "start_location_y",
        "end_location_x",
        "end_location_y",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def add_ride_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add legacy analysis columns without row-wise apply or duration string parsing."""
    result = data.copy()
    result["duration"] = result["end_time"] - result["start_time"]
    duration_seconds = result["duration"].dt.total_seconds()
    result["ttl_min"] = duration_seconds / 60.0

    # The legacy fields are retained.  Invalid/negative durations remain nullable,
    # rather than being incorrectly parsed from a string representation.
    valid_duration = result["duration"].where(duration_seconds.ge(0))
    components = valid_duration.dt.components
    result["dur_day"] = components["days"].astype("Int64")
    result["dur_hr"] = components["hours"].astype("Int64")
    result["dur_min"] = components["minutes"].astype("Int64")
    result["dur_sec"] = components["seconds"].astype("Int64")

    # start_time is parsed as supplied.  Do not silently reinterpret naive local
    # Shanghai timestamps as UTC (the old utctimetuple call was ambiguous).
    weekday = result["start_time"].dt.isocalendar().day.astype("Int64")
    hour_zero_based = result["start_time"].dt.hour.astype("Int64")
    result["dayId"] = weekday
    result["dayType"] = np.where(
        weekday.isin([6, 7]), "weekends", np.where(weekday.notna(), "weekdays", pd.NA)
    )
    # The original output shifts hours to 1--24; retain that external convention.
    result["hourId"] = hour_zero_based + 1
    result["hourType"] = np.where(
        hour_zero_based.between(7, 8) | hour_zero_based.between(17, 20),
        "rush hours",
        np.where(hour_zero_based.notna(), "non-rush hours", pd.NA),
    )

    result["distance"] = np.round(
        haversine_km(
            result["start_location_x"], result["start_location_y"],
            result["end_location_x"], result["end_location_y"],
        ),
        3,
    )
    result["disToCenter"] = np.round(
        haversine_km(
            result["start_location_x"], result["start_location_y"],
            SHANGHAI_CENTER[0], SHANGHAI_CENTER[1],
        ),
        3,
    )
    return result


def write_legacy_summaries(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Write the original four CSV outputs without temporary CSV round trips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        data.groupby("ttl_min", dropna=True).size().rename("timeNum").reset_index(name="timeNum")
        .rename(columns={"ttl_min": "time"})
        .sort_values("time")
        .to_csv(output_dir / "gbtimeListToDF.csv", index=False)
    )
    (
        data.groupby("hourId", dropna=True).size().rename("orderid").reset_index()
        .to_csv(output_dir / "hour_num_df.csv", index=False)
    )
    (
        data.groupby("distance", dropna=True).size().rename("distanceNum").reset_index()
        .sort_values("distanceNum")
        .to_csv(output_dir / "gbDisListToDF.csv", index=False)
    )

    locations = data.loc[
        data["end_location_x"].notna() & data["end_location_y"].notna(),
        ["end_location_x", "end_location_y"],
    ].rename(columns={"end_location_x": "longitude", "end_location_y": "latitude"})
    locations.to_csv(output_dir / "locDisDF.csv", index=False)
    return locations.reset_index(drop=True)


def to_local_km(locations: pd.DataFrame) -> tuple[np.ndarray, tuple[float, float]]:
    """Equirectangular local projection, suitable for short Shanghai distances."""
    origin_lon = float(locations["longitude"].median())
    origin_lat = float(locations["latitude"].median())
    latitude_scale = 110.574
    longitude_scale = 111.320 * np.cos(np.deg2rad(origin_lat))
    # coordinates = locations[["longitude", "latitude"]].to_numpy(dtype=float)
    coordinates = locations[["longitude", "latitude"]].to_numpy(dtype=float, copy=True)
    coordinates[:, 0] = (coordinates[:, 0] - origin_lon) * longitude_scale
    coordinates[:, 1] = (coordinates[:, 1] - origin_lat) * latitude_scale
    return coordinates, (origin_lon, origin_lat)


def from_local_km(points: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    origin_lon, origin_lat = origin
    latitude_scale = 110.574
    longitude_scale = 111.320 * np.cos(np.deg2rad(origin_lat))
    converted = points.copy()
    converted[:, 0] = converted[:, 0] / longitude_scale + origin_lon
    converted[:, 1] = converted[:, 1] / latitude_scale + origin_lat
    return converted


def save_k_distance_plot(distances_km: np.ndarray, path: Path, neighbors: int) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(np.sort(distances_km), linewidth=1)
    axis.set(title=f"{neighbors}-nearest-neighbour distance", xlabel="Sorted point index", ylabel="Distance (km)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cluster_locations(locations: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    if locations.empty:
        print("No valid end-location coordinates: clustering outputs were skipped.")
        return

    count = len(locations)
    local_km, origin = to_local_km(locations)
    radians = np.deg2rad(locations[["latitude", "longitude"]].to_numpy(dtype=float))
    neighbors = min(max(args.knn_neighbors, 1), count)
    min_samples = min(max(args.dbscan_min_samples, 1), count)

    # Haversine KNN makes the elbow scale compatible with DBSCAN eps in kilometres.
    knn = NearestNeighbors(n_neighbors=neighbors, metric="haversine", algorithm="ball_tree")
    knn_distances, _ = knn.fit(radians).kneighbors(radians)
    k_distance_km = knn_distances[:, neighbors - 1] * EARTH_RADIUS_KM
    save_k_distance_plot(k_distance_km, output_dir / "knn_k_distance.png", neighbors)

    clusters = min(max(args.kmeans_clusters, 1), count)
    kmeans = KMeans(
        n_clusters=clusters,
        n_init=args.kmeans_n_init,
        random_state=args.random_state,
    ).fit(local_km)
    centers_km = kmeans.cluster_centers_

    # "Within the annulus of any centre" is equivalent to checking the nearest
    # centre, and replaces the original O(number_of_points * number_of_centres)
    # nested Python loops.
    nearest_center_distance, _ = NearestNeighbors(n_neighbors=1).fit(centers_km).kneighbors(local_km)
    nearest_center_distance = nearest_center_distance[:, 0]
    border_mask = np.abs(nearest_center_distance - args.border_radius_km) <= args.border_tolerance_km
    locations.loc[border_mask, ["longitude", "latitude"]].drop_duplicates().to_csv(
        output_dir / "borderPointsDF.csv", index=False
    )
    pd.DataFrame(
        from_local_km(centers_km, origin), columns=["longitude", "latitude"]
    ).assign(size=pd.Series(kmeans.labels_).value_counts().sort_index().to_numpy()).to_csv(
        output_dir / "kmeans_centers.csv", index=False
    )

    dbscan = DBSCAN(
        eps=args.dbscan_eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit(radians)
    labels = dbscan.labels_
    clustered = labels != -1
    distinct_clusters = np.unique(labels[clustered])
    score = np.nan
    if len(distinct_clusters) >= 2:
        score = davies_bouldin_score(radians[clustered], labels[clustered])
    (
        pd.Series(labels).value_counts().sort_index().rename_axis("cluster").rename("count")
        .to_csv(output_dir / "dbscan_cluster_counts.csv")
    )
    locations.assign(dbscan_label=labels).to_csv(output_dir / "dbscan_locations.csv", index=False)

    fig, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(locations["longitude"], locations["latitude"], c=labels, s=4, cmap="tab20")
    axis.set(title="DBSCAN parking-location clusters", xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(scatter, ax=axis, label="DBSCAN label (-1 = noise)")
    fig.tight_layout()
    fig.savefig(output_dir / "dbscan_clusters.png", dpi=160)
    plt.close(fig)

    print(
        f"Locations: {count}; KNN neighbours: {neighbors}; DBSCAN clusters: {len(distinct_clusters)}; "
        f"noise points: {(labels == -1).sum()}; Davies-Bouldin (noise excluded): {score}"
    )


def main() -> None:
    args = parse_args()
    if args.dbscan_eps_km <= 0 or args.border_radius_km < 0 or args.border_tolerance_km < 0:
        raise ValueError("Distance parameters must be non-negative; DBSCAN eps must be positive.")
    data = validate_and_load(args.input)
    invalid_times = data[["start_time", "end_time"]].isna().any(axis=1).sum()
    negative_durations = (data["end_time"] < data["start_time"]).sum()
    if invalid_times or negative_durations:
        print(f"Warning: invalid timestamps={invalid_times}, negative durations={negative_durations}.")
    enriched = add_ride_features(data)
    locations = write_legacy_summaries(enriched, args.output_dir)
    enriched.to_csv(args.output_dir / "mobike_enriched.csv", index=False)
    cluster_locations(locations, args, args.output_dir)


if __name__ == "__main__":
    main()
