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
