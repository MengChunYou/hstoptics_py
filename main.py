# main.py
from src.generate_simulated_data import generate_simulated_data
from src.make_plots import make_plots


def main():
    generate_simulated_data()
    make_plots()


if __name__ == "__main__":
    main()
