# generate_cluster_results.py
import os
import logging
import pandas as pd
from src.clustering_algorithms.stdbscan import STDBSCAN
from src.clustering_algorithms.hstoptics import HSTOPTICS
from src.utils import read_all_data
from src.parameters import param_sets
from src.generate_plots import generate_reachability_plot_with_faults


def write_cluster_result(
    result_df: pd.DataFrame,
    base_file_name: str,
    data_dim: int,
    algorithm_name: str,
    param_order: int = 1,
    dir_path: str = 'outputs/cluster_results/'
) -> None:
    """
    Write cluster results to a file.

    Args:
        result_df: The DataFrame containing clustering results.
        base_file_name: base_file_name is like 'clustering_structure_1'.
        data_dim: The dimensionality of the data.
            - `2` for 2D data (x, y)
            - `3` for 3D data (x, y, t)
        algorithm_name: The name of the clustering algorithm used.
        param_order: The order of parameters used for clustering.
        dir_path: The directory path where the output file will be saved.
    """
    file_name = f"{base_file_name}_{data_dim}d_cluster_result_{algorithm_name}_param{param_order}.csv"
    file_path = os.path.join(f"{dir_path}{data_dim}d/", file_name)

    # Ensure that all values in the 'cluster' column are integers
    result_df['cluster'] = result_df['cluster'].astype(int)

    os.makedirs(f"{dir_path}{data_dim}d/", exist_ok=True)
    result_df.to_csv(file_path, index=False)

    logging.info(f"Save cluster result: {file_path}")


def generate_stdbscan_cluster_results(
    data_df: pd.DataFrame,
    data_dim: int,
    spatial_threshold: float,
    temporal_threshold: float,
    min_neighbors: int
) -> pd.DataFrame:
    """
    Apply ST-DBSCAN clustering algorithm on the provided data and return the clustering results.

    This function uses the ST-DBSCAN algorithm to cluster data based on spatial and temporal thresholds.
    If the data is 2D, temporal column ('t') is set as 0.

    Args:
        data_df: Input DataFrame containing the data to be clustered.
        data_dim: The dimensionality of the data.
        spatial_threshold: Maximum distance between points in the spatial dimension.
        temporal_threshold: Maximum time difference between points in the temporal dimension.
        min_neighbors: Minimum number of points required to form a cluster.

    Returns:
        result_df: The original DataFrame with an added 'cluster' column that indicates the cluster label
                   for each point. Points that are classified as noise will have a cluster label of -1.
    """
    data_df_copy = data_df.copy()

    # If the data is 2D, set 't' column as 0
    if data_dim == 2:
        data_df_copy['t'] = 0

    # Create an instance of the STDBSCAN algorithm
    st_dbscan = STDBSCAN(
        spatial_threshold,
        temporal_threshold,
        min_neighbors
    )

    # Apply the STDBSCAN algorithm to the data
    result_df = st_dbscan.fit_transform(
        df=data_df_copy,
        col_lat="x",
        col_lon="y",
        col_time="t"
    )

    # Return the DataFrame with clustering results
    return result_df


def generate_hstoptics_cluster_results(
    data_df: pd.DataFrame,
    base_file_name: str,
    data_dim: int,
    eps_s: float,
    eps_t: float,
    min_pts: int,
    w_s: float,
    w_t: float,
    diff: float,
    window_size: int
) -> pd.DataFrame:
    """
    Apply HST-OPTICS clustering algorithm on the provided data and return the clustering results.
    If the data is 2D, temporal column ('t') is set as 0.

    Args:
        data_df: Input DataFrame containing the data to be clustered.
        base_file_name: base_file_name is like 'clustering_structure_1'.
        data_dim: The dimensionality of the data.
        eps_s: Radius of the spatial epsilon neighborhood.
        eps_t: Radius of the temporal epsilon neighborhood.
        min_pts: Number of minimum points required in the eps_s and eps_t
                 neighborhood for core points (including the point itself).
        w_s: Weight of spatial distance.
        w_t: Weight of temporal distance.
        diff: Threshold for detecting a significant change (fault) in reachability score.
        window_size: Number of following points to consider for slope calculation.

    Returns:
        result_df: The original DataFrame with an added 'cluster' column that indicates the cluster label
                   for each point. Points that are classified as noise will have a cluster label of -1.
    """
    data_df_copy = data_df.copy()

    # If the data is 2D, set 't' column as 0
    if data_dim == 2:
        data_df_copy['t'] = 0

    # Create an instance of the STDBSCAN algorithm
    hst_optics = HSTOPTICS(eps_s, eps_t, min_pts)

    # Compute reachability score
    hst_optics.compute_order_and_reachability(
        data_df=data_df_copy,
        xyt_colnames=['x', 'y', 't'],
        w_s=w_s,
        w_t=w_t
    )

    # Early return if there is no finite value in reachability score
    if hst_optics.validate_reachability_scores():
        result_df = data_df_copy
        result_df['cluster'] = -1
        hst_optics.cluster_result = result_df
        return result_df

    # Identifies significant changes (faults).
    hst_optics.find_faults(diff, window_size)

    # Generate and save a reachability plot with fault lines.
    generate_reachability_plot_with_faults(
        hst_optics.cluster_profile['reach_score'],
        hst_optics.cluster_profile['fault'],
        base_file_name,
        data_dim
    )

    # Assigns hierarchical density levels.
    hst_optics.assign_density_levels()

    # Get clustering result
    hst_optics.assign_clusters()
    result_df = hst_optics.cluster_result
    return result_df


def generate_cluster_results():
    data_df_list, base_file_name_list = read_all_data(data_dir="simulated_data/")

    for data_df, base_file_name in zip(
        data_df_list,
        base_file_name_list
    ):
        for data_dim in [2, 3]:
            # Generate and write cluster result using stdbscan algorithm
            # algorithm_name = "stdbscan"
            # kwargs = param_sets[algorithm_name][base_file_name][data_dim]

            # cluster_result = generate_stdbscan_cluster_results(data_df, data_dim, **kwargs)
            # write_cluster_result(
            #     cluster_result,
            #     base_file_name,
            #     data_dim,
            #     algorithm_name=algorithm_name
            # )

            # Generate and write cluster result using hstoptics algorithm
            algorithm_name = "hstoptics"
            kwargs = param_sets[algorithm_name][base_file_name][data_dim]

            cluster_result = generate_hstoptics_cluster_results(data_df, base_file_name, data_dim, **kwargs)
            write_cluster_result(
                cluster_result,
                base_file_name,
                data_dim,
                algorithm_name=algorithm_name,
            )
