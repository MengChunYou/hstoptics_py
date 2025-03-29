# generate_simulated_data.py
import os
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple


def write_simulated_data(
    simulated_data: pd.DataFrame,
    file_order: int,
    dir_path: str = "simulated_data"
):
    """
    Write simulated data

    Args:
        simulated_data: The data to write.
        file_order: The file order number.
        dir_path: The directory to store the file.
    """
    os.makedirs(dir_path, exist_ok=True)

    file_name = f"clustering_structure_{file_order}.csv"
    file_path = os.path.join(dir_path, file_name)

    simulated_data.to_csv(file_path, index=False)

    logging.info(f"Simulated data written to {file_path}")


def density_to_n_points(
    density: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    t_range: Tuple[float, float]
) -> int:
    """
    Calculates the number of points to generate
    based on the given density and ranges for x, y, and t.

    Args:
        density: The density factor for the points generation.
        x_range: The range for the x values (min, max).
        y_range: The range for the y values (min, max).
        t_range: The range for the t values (min, max).

    Returns:
        n_points: The total number of points to generate.
    """
    n_points = density * \
        (x_range[1] - x_range[0]) * \
        (y_range[1] - y_range[0]) * \
        (t_range[1] - t_range[0])

    return int(n_points)


def generate_noise_data(
    n_points: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    t_range: Tuple[float, float]
) -> pd.DataFrame:
    """
    Generates random noise data within the specified ranges for x, y, and t.

    Args:
        n_points: Number of noise points to generate.
        x_range: Range for the x values (min, max).
        y_range: Range for the y values (min, max).
        t_range: Range for the t values (min, max).

    Returns:
        simulated_data: DataFrame containing the generated noise data.
    """
    np.random.seed(123)

    noise_data = pd.DataFrame({
        "x": np.random.uniform(x_range[0], x_range[1], n_points),
        "y": np.random.uniform(y_range[0], y_range[1], n_points),
        "t": np.random.uniform(t_range[0], t_range[1], n_points)
    })

    return noise_data


def generate_cluster_data(
    density: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    t_range: Tuple[float, float]
) -> pd.DataFrame:
    """
    Generates simulated clustered data based on density and ranges.

    Args:
        density: The density factor to scale the number of points in cluster.
        x_range: The range for the x values (min, max).
        y_range: The range for the y values (min, max).
        t_range: The range for the t values (min, max).

    Returns:
        cluster_data: DataFrame containing the generated clustered data.
    """
    n_points = density_to_n_points(density, x_range, y_range, t_range)

    np.random.seed(123)

    # Generate clustered data within the specified ranges
    cluster_data = pd.DataFrame({
        "x": np.random.uniform(x_range[0], x_range[1], n_points),
        "y": np.random.uniform(y_range[0], y_range[1], n_points),
        "t": np.random.uniform(t_range[0], t_range[1], n_points)
    })

    return cluster_data


def filter_cluster_data(
    cluster_data: pd.DataFrame,
    restriction: Tuple[float, float]
) -> pd.DataFrame:
    """
    Filters the given cluster_data by dropping rows
    where x_t (x - t) is not within the specified range.

    Args:
        cluster_data: The DataFrame containing the cluster data.
        restriction: The range for x_t.

    Returns:
        filtered_data: The filtered DataFrame.
    """
    # Calculate x_t
    cluster_data['x_t'] = cluster_data['x'] - cluster_data['t']

    # Filter the data based on the restriction range
    filtered_data = cluster_data[
        (cluster_data['x_t'] >= restriction[0]) &
        (cluster_data['x_t'] <= restriction[1])
    ]

    return filtered_data.drop(columns=['x_t'])


def generate_clustering_structure(
    noise_data: pd.DataFrame,
    density_list: List[int],
    x_range_list: List[Tuple[float, float]],
    y_range_list: List[Tuple[float, float]],
    t_range_list: List[Tuple[float, float]],
    restriction_list: List[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Generates simulated data with a clustering structure
    by combining noise points
    with clusters generated within the specified ranges and densities.

    Args:
        noise_data: DataFrame of noise points.
        density_list: List containing the density of points in each cluster.
        x_range_list: List of tuples representing the range for x values of clusters.
        y_range_list: List of tuples representing the range for y values of clusters.
        t_range_list: List of tuples representing the range for t values of clusters.
        restriction_list: List of tuples (or None) representing the range for x_t.
    Returns:
        simulated_data: DataFrame with the simulated data, combining noise and clusters.
    """
    simulated_data = noise_data.copy()

    np.random.seed(123)

    if restriction_list is None:
        restriction_list = [None] * len(density_list)

    for density, x_range, y_range, t_range, restriction in zip(
        density_list, x_range_list, y_range_list, t_range_list, restriction_list
    ):
        # Generate cluster data
        cluster_data = generate_cluster_data(
            density, x_range, y_range, t_range
        )

        # Filter cluster data if restriction is provided
        if restriction is not None:
            cluster_data = filter_cluster_data(cluster_data, restriction)

        # Combine with the simulated data
        simulated_data = pd.concat([simulated_data, cluster_data], ignore_index=True)

    return simulated_data


def generate_simulated_data():
    # Set up logging to capture function execution details
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Generate and write noise data (no clusters)
    noise_data = generate_noise_data(
        n_points=30,
        x_range=(-9.9, 9.9),
        y_range=(-9.9, 9.9),
        t_range=(-9.9, 9.9)
    )
    write_simulated_data(noise_data, 1)

    # Generate and write data with one cluster and noise
    simulated_data = generate_clustering_structure(
        noise_data=noise_data,
        density_list=[50],
        x_range_list=[(-1, 1)],
        y_range_list=[(-2, 2)],
        t_range_list=[(-1, 1)]
    )
    write_simulated_data(simulated_data, 2)

    # Generate and write data with two clusters and noise
    simulated_data = generate_clustering_structure(
        noise_data=noise_data,
        density_list=[50, 50],
        x_range_list=[(-1 - 2, 1 - 2), (-2 + 2, 2 + 2)],
        y_range_list=[(-2 - 2, 2 - 2), (-1 + 2, 1 + 2)],
        t_range_list=[(-1 - 2, 1 - 2), (-1 + 2, 1 + 2)]
    )
    write_simulated_data(simulated_data, 3)

    # Generate and write data with one low-density cluster, one high-density cluster, and noise
    simulated_data = generate_clustering_structure(
        noise_data=noise_data,
        density_list=[5, 50],
        x_range_list=[(-5, 5), (-1, 1)],
        y_range_list=[(-3, 3), (-1, 1)],
        t_range_list=[(-3, 3), (-2, 2)]
    )
    write_simulated_data(simulated_data, 4)

    # Generate and write data with one low-density cluster, two high-density clusters, and noise
    simulated_data = generate_clustering_structure(
        noise_data=noise_data,
        density_list=[5, 50, 50],
        x_range_list=[(-5, 5), (-3, -1), (1, 3)],
        y_range_list=[(-3, 3), (-1, 1), (-1, 1)],
        t_range_list=[(-3, 3), (-2, 2), (-2, 2)]
    )
    write_simulated_data(simulated_data, 5)

    # Generate and write data with one low-density cluster, one medium-density cluster, one high-density cluster, and noise
    simulated_data = generate_clustering_structure(
        noise_data=noise_data,
        density_list=[5, 10, 100],
        x_range_list=[(-5, 5), (-2.5, 2.5), (-1, 1)],
        y_range_list=[(-3, 3), (-2, 2), (-1, 1)],
        t_range_list=[(-3, 3), (-2, 2), (-1, 1)]
    )
    write_simulated_data(simulated_data, 6)

    # Generate and write data with one low-density cluster, one high-density cluster, noise, and moving x-range
    simulated_data = generate_clustering_structure(
        noise_data=noise_data,
        density_list=[5, 50],
        x_range_list=[(-5, 5), (-3, 3)],
        y_range_list=[(-3, 3), (-1, 1)],
        t_range_list=[(-3, 3), (-2, 2)],
        restriction_list=[(-3, 3), (-1, 1)],
    )
    write_simulated_data(simulated_data, 7)
