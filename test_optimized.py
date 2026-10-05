"""Standalone DBSCAN validation for Mobike end-location data.

This is an exploratory test script, not a pytest test.  It validates a chosen
DBSCAN parameter set and exports the labels for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


EARTH_RADIUS_KM = 6371.0088
REQUIRED_COLUMNS = {"longitude", "latitude"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Mobike DBSCAN clustering parameters.")
    parser.add_argument("--input", type=Path, default=Path("data/locDisDF.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--neighbors", type=int, default=200, help="KNN reference neighbour count.")
    parser.add_argument("--eps-km", type=float, default=0.30, help="DBSCAN neighbourhood radius in km.")
    parser.add_argument("--min-samples", type=int, default=200, help="Points required for a DBSCAN core point.")
    return parser.parse_args()


def load_locations(path: Path) -> pd.DataFrame:
    """Read valid coordinate pairs only, so an invalid row cannot break clustering."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {path.resolve()}")
    locations = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(locations.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    locations = locations.loc[:, ["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    if locations.empty:
        raise ValueError("There are no valid longitude/latitude rows to cluster.")
    return locations.reset_index(drop=True)


def save_k_distance_plot(radians: np.ndarray, neighbours: int, output: Path) -> None:
    """Draw distance to the nth nearest neighbour, expressed in kilometres."""
    model = NearestNeighbors(n_neighbors=neighbours, metric="haversine", algorithm="ball_tree")
    distances, _ = model.fit(radians).kneighbors(radians)
    kth_distance_km = np.sort(distances[:, neighbours - 1] * EARTH_RADIUS_KM)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(kth_distance_km, linewidth=1)
    axis.set(title=f"Distance to {neighbours}th nearest neighbour", xlabel="Sorted point index", ylabel="Distance (km)")
    figure.tight_layout()
    figure.savefig(output / "test_knn_k_distance.png", dpi=160)
    plt.close(figure)


def dbscan_roles(labels: np.ndarray, core_indices: np.ndarray) -> np.ndarray:
    """Return a readable role for every point using DBSCAN's official core indices."""
    roles = np.full(labels.size, "border", dtype=object)
    roles[labels == -1] = "noise"
    roles[core_indices] = "core"
    return roles


def main() -> None:
    args = parse_args()
    if args.eps_km <= 0 or args.neighbors < 1 or args.min_samples < 1:
        raise ValueError("eps-km, neighbors and min-samples must all be positive.")
    locations = load_locations(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    point_count = len(locations)
    neighbours = min(args.neighbors, point_count)
    min_samples = min(args.min_samples, point_count)
    # scikit-learn Haversine expects [latitude, longitude] in radians, not degrees.
    radians = np.deg2rad(locations[["latitude", "longitude"]].to_numpy(dtype=float))
    save_k_distance_plot(radians, neighbours, args.output_dir)

    model = DBSCAN(
        eps=args.eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit(radians)
    labels = model.labels_
    roles = dbscan_roles(labels, model.core_sample_indices_)
    clustered = labels != -1
    cluster_labels = np.unique(labels[clustered])
    score = np.nan
    if cluster_labels.size >= 2:
        # Noise is not a real cluster, so exclude it from this quality reference.
        score = davies_bouldin_score(radians[clustered], labels[clustered])

    result = locations.assign(dbscan_label=labels, dbscan_role=roles)
    result.to_csv(args.output_dir / "test_dbscan_point_roles.csv", index=False)
    (
        pd.Series(labels).value_counts().sort_index().rename_axis("cluster").rename("count")
        .to_csv(args.output_dir / "test_dbscan_cluster_counts.csv")
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    colour = {"core": "#1f77b4", "border": "#ff7f0e", "noise": "#bdbdbd"}
    for role in ("noise", "border", "core"):
        subset = result["dbscan_role"] == role
        axis.scatter(result.loc[subset, "longitude"], result.loc[subset, "latitude"], s=3, c=colour[role], label=role, alpha=0.55)
    axis.set(title="DBSCAN point roles", xlabel="Longitude", ylabel="Latitude")
    axis.legend(markerscale=3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "test_dbscan_roles.png", dpi=160)
    plt.close(figure)

    role_counts = pd.Series(roles).value_counts()
    print(f"Points: {point_count}")
    print(f"Parameters: eps={args.eps_km:.3f} km, min_samples={min_samples}, KNN neighbours={neighbours}")
    print(f"DBSCAN clusters: {cluster_labels.size}; Davies-Bouldin (noise excluded): {score}")
    print(f"Core: {role_counts.get('core', 0)}; border: {role_counts.get('border', 0)}; noise: {role_counts.get('noise', 0)}")


if __name__ == "__main__":
    main()
