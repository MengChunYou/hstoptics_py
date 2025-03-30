# utils.py
import os
import pandas as pd
from typing import List


def read_all_simulated_data(data_dir: str = 'simulated_data/') -> List[pd.DataFrame]:
    """
    Read all simulated data and store into a list.

    Args:
        data_dir: The directory to read the files.

    Returns:
        data_df_list: A list of DataFrame objects. Each DataFrame contains simulated data with x, y, and t.
        base_file_name_list: base_file_name is like 'clustering_structure_1'.
    """
    data_df_list = []
    base_file_name_list = []

    for file_name in os.listdir(data_dir):
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
