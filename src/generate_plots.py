# generate_plots.py
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import read_all_data
from typing import Any, List, Dict


def get_color_map(
    labels: List[Any],
    colors: List[str] = [
        "gray", "blue", "green", "red", "yellow",
        "magenta", "cyan", "darkgray", "brown", "purple"
    ]
) -> dict[Any, str]:
    """
    Generate a mapping of unique labels to colors.

    Args:
        labels: A list of unique labels (e.g., cluster labels).
        colors: A predefined list of colors.

    Returns:
        A dictionary mapping each label to a color.
    """
    return {label: colors[i % len(colors)] for i, label in enumerate(labels)}


def generate_2d_plot(
    df: pd.DataFrame,
    base_file_name: str,
    color_by_cluster: bool = False
) -> None:
    """
    Generate and save a 2D scatter plot from the given DataFrame.

    Args:
        df: DataFrame containing 'x' and 'y' columns.
        base_file_name: Base name of the source csv file.
        color_by_cluster: If True, color points by 'cluster' column.
    """
    plot_type = "cluster_plot" if color_by_cluster else "original_plot"
    dir_path = f"outputs/{plot_type}s/2d/"
    os.makedirs(dir_path, exist_ok=True)

    plt.figure(figsize=(8, 6))

    if color_by_cluster:
        output_path = os.path.join(dir_path, f"{base_file_name}_plot.png")
        cluster_labels = df['cluster'].unique()
        color_map = get_color_map(cluster_labels)

        for cluster_label in cluster_labels:
            cluster_data = df[df['cluster'] == cluster_label]
            plt.scatter(
                cluster_data['x'], cluster_data['y'],
                alpha=0.3, color=color_map[cluster_label], s=10,
                label = "noise" if (cluster_label == -1) else f"{cluster_label}"
            )
        plt.legend(title="Cluster")
        plt.title(f"2D Scatter Plot for {base_file_name}\n(Color by Cluster)")
    else:
        output_path = os.path.join(dir_path, f"{base_file_name}_2d_plot.png")
        plt.scatter(df["x"], df["y"], alpha=0.3, color="gray", s=10)
        plt.title(f"2D Scatter Plot for {base_file_name}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, which="both")

    plt.savefig(output_path)
    plt.close()

    logging.info(f"Save 2D plot: {output_path}")


def generate_3d_plot(
    df: pd.DataFrame,
    base_file_name: str,
    color_by_cluster: bool = False
) -> None:
    """
    Generate and save a 3D scatter plot from the given DataFrame.

    Args:
        df: DataFrame containing 'x', 'y', and 't' columns.
        base_file_name: Base name of the source csv file.
        color_by_cluster: If True, color points by 'cluster' column.
    """
    plot_type = "cluster_plot" if color_by_cluster else "original_plot"
    dir_path = f"outputs/{plot_type}s/3d/"
    os.makedirs(dir_path, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    if color_by_cluster:
        output_path = os.path.join(dir_path, f"{base_file_name}_plot.png")
        cluster_labels = df['cluster'].unique()
        color_map = get_color_map(cluster_labels)

        for cluster_label in cluster_labels:
            cluster_data = df[df['cluster'] == cluster_label]
            ax.scatter(
                cluster_data["x"], cluster_data["y"], cluster_data["t"],
                alpha=0.3, color=color_map[cluster_label], s=5,
                label = "noise" if (cluster_label == -1) else f"{cluster_label}"
            )
        ax.legend(loc="upper left", title="Cluster")
        ax.set_title(f"3D Scatter Plot for {base_file_name}\n(Colored by Cluster)")
    else:
        output_path = os.path.join(dir_path, f"{base_file_name}_3d_plot.png")
        ax.scatter(df["x"], df["y"], df["t"], alpha=0.3, color="gray", s=5)
        ax.set_title(f"3D Scatter Plot for {base_file_name}")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.grid(True)
    ax.view_init(elev=15, azim=-75)

    plt.savefig(output_path)
    plt.close()

    logging.info(f"Save 3D plot: {output_path}")


def generate_original_plots(
    data_dir: str = 'simulated_data/'
) -> None:
    """
    Generate and save 2D and 3D original plots for all files in the data_dir.

    Args:
        data_dir: The directory to read the files.
    """
    data_df_list, base_file_name_list = read_all_data(data_dir)
    for simulated_data, base_file_name in zip(
        data_df_list,
        base_file_name_list
    ):
        generate_2d_plot(simulated_data, base_file_name)
        generate_3d_plot(simulated_data, base_file_name)


def generate_cluster_plots(
    data_dir: str = "outputs/cluster_results"
) -> None:
    """
    Generate and save 2D and 3D cluster plots for all files in the data_dir.

    Args:
        data_dir: The directory to read the files.
    """
    for data_dim in [2, 3]:
        data_df_list, base_file_name_list = read_all_data(
            os.path.join(data_dir, f"{data_dim}d")
        )
        for simulated_data, base_file_name in zip(
            data_df_list,
            base_file_name_list
        ):
            if data_dim == 2:
                generate_2d_plot(simulated_data, base_file_name, color_by_cluster=True)
            elif data_dim == 3:
                generate_3d_plot(simulated_data, base_file_name, color_by_cluster=True)
