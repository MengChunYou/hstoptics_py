# utils.py
import os
import pandas as pd
import numpy as np
from typing import List


def read_all_data(data_dir: str) -> List[pd.DataFrame]:
    """
    Read all data in dir and store into a list.

    Args:
        data_dir: The directory to read the files.

    Returns:
        data_df_list: A list of DataFrame objects.
        base_file_name_list: base_file_name is base name of the source file, without extension.
    """
    data_df_list = []
    base_file_name_list = []

    for file_name in sorted(os.listdir(data_dir)):
        base_file_name = os.path.splitext(file_name)[0]
        data_path = os.path.join(data_dir, file_name)
        simulated_data = pd.read_csv(data_path)

        data_df_list.append(simulated_data)
        base_file_name_list.append(base_file_name)

    return data_df_list, base_file_name_list


def read_simulated_data(
    clustering_structure_order: int,
    data_dir: str = "simulated_data/"
) -> pd.DataFrame:
    """
    Read specific simulated data.

    Args:
        clustering_structure_order: from 1 to 7
        data_dir: The directory to read the file.

    Return:
        simulated_data: DataFrame contains simulated data with x, y, and t.
    """
    data_path = os.path.join(
        data_dir,
        f"clustering_structure_{clustering_structure_order}.csv"
    )
    simulated_data = pd.read_csv(data_path)

    return simulated_data


def get_distance(
    one_point: pd.Series,
    other_points: pd.DataFrame,
    xy_colnames: List[str] = ['x', 'y']
) -> pd.Series:
    """
    Compute spatial Euclidean distance between one point and multiple other points.

    Args:
        one_point: A single point, usually a row of a DataFrame.
        other_points: A DataFrame containing multiple points to compare to.
        xy_colnames: List of column names for ['x', 'y'].

    Returns:
        A Series of spatial distances from the given point to each of the other points.
    """
    x_col, y_col = xy_colnames[0], xy_colnames[1]
    return np.sqrt(
        (other_points[x_col] - one_point[x_col]) ** 2 +
        (other_points[y_col] - one_point[y_col]) ** 2
    )


def get_st_distance(
    one_point: pd.Series,
    other_points: pd.DataFrame,
    weight_s: float = 1,
    weight_t: float = 1,
    xyt_colnames: List[str] = ['x', 'y', 't']
) -> pd.Series:
    """
    Compute spatio-temporal distance between one point and multiple other points.

    Args:
        one_point: A single point, usually a row of a DataFrame.
        other_points: A DataFrame containing multiple points to compare to.
        weight_s: Weight for spatial distance.
        weight_t: Weight for temporal distance.
        xyt_colnames: List of column names for ['x', 'y', 't'].

    Returns:
        A Series of spatio-temporal distances from the given point to each of the other points.
    """
    x_col, y_col, t_col = xyt_colnames[0], xyt_colnames[1], xyt_colnames[2]
    s_dist = get_distance(one_point, other_points, [x_col, y_col])
    t_dist = np.abs(other_points[t_col] - one_point[t_col])
    sq_st_dist = (weight_s * s_dist) ** 2 + (weight_t * t_dist) ** 2
    return np.sqrt(sq_st_dist)
