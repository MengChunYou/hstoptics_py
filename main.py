# main.py
import logging
from src.generate_simulated_data import generate_simulated_data
from src.generate_plots import generate_original_plots, generate_cluster_plots
from src.generate_cluster_results import generate_cluster_results


def main():
    # Set up logging to capture function execution details
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Generate simulated data
    generate_simulated_data()

    # Generate original plots based on the simulated data (without clustering)
    generate_original_plots()

    # Generate clustering results
    generate_cluster_results()

    # Generate plots that color the data points by cluster labels
    generate_cluster_plots()

if __name__ == "__main__":
    main()
