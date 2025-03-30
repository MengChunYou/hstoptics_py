# main.py
import logging
from src.generate_simulated_data import generate_simulated_data
from src.make_plots import make_plots
from src.generate_cluster_results import generate_cluster_results


def main():
    # Set up logging to capture function execution details
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    generate_simulated_data()
    make_plots()
    generate_cluster_results()


if __name__ == "__main__":
    main()
