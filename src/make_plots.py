# make_plots.py
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import read_all_simulated_data


def make_2d_plot(
    df: pd.DataFrame,
    base_file_name: str,
    dir_path: str = "outputs/original_plots/2d/"
) -> None:
    """
    Generate and save a 2D scatter plot of from the given DataFrame.

    Args:
        df: DataFrame containing 'x' and 'y' columns.
        base_file_name: Base name of the source csv file.
        dir_path: The directory to store the file.
    """
    os.makedirs(dir_path, exist_ok=True)
    output_path = os.path.join(dir_path, f"{base_file_name}_2d_plot.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["x"], df["y"], alpha=0.3, color="gray", s=10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2D Scatter Plot for {base_file_name}")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, which="both")

    plt.savefig(output_path)
    plt.close()

    logging.info(f"Save 2D plot: {output_path}")


def make_3d_plot(
    df: pd.DataFrame,
    base_file_name: str,
    dir_path: str = "outputs/original_plots/3d/"
) -> None:
    """
    Generate and save a 3D scatter plot from the given DataFrame.

    Args:
        df: DataFrame containing 'x', 'y', and 't' columns.
        base_file_name: Base name of the source csv file.
        dir_path: The directory to store the file.
    """
    os.makedirs(dir_path, exist_ok=True)
    output_path = os.path.join(dir_path, f"{base_file_name}_3d_plot.png")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df["x"], df["y"], df["t"], alpha=0.3, color="gray", s=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"3D Scatter Plot for {base_file_name}")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.grid(True)
    ax.view_init(elev=15, azim=-75)

    plt.savefig(output_path)
    plt.close()

    logging.info(f"Save 3D plot: {output_path}")


def make_plots(data_dir: str = 'simulated_data/'):
    """
    Generate and save 2D and 3D plots for all files in the data_dir.

    Args:
        data_dir: The directory to read the files.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data_df_list, base_file_name_list = read_all_simulated_data(data_dir)

    for simulated_data, base_file_name in zip(
        data_df_list,
        base_file_name_list
    ):
        make_2d_plot(simulated_data, base_file_name)
        make_3d_plot(simulated_data, base_file_name)
