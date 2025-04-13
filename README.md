# HST-OPTICS (Python Version)

This repository contains the **Python implementation** of the [HST-OPTICS algorithm](https://github.com/MengChunYou/hstoptics), originally developed in R for a master's thesis titled _"An OPTICS-based Algorithm for Identifying Spatio-Temporal Density Faults in Hierarchical Clustering Structures."_

## Table of Contents

- [Project Abstract](#project-abstract)
- [Repository Structure](#repository-structure)

## Project Abstract

The HST-OPTICS algorithm improves upon previous spatio-temporal clustering methods by identifying density faults in hierarchical clusters. This approach reveals complete clustering structures, including density differences and hierarchical relationships previously undetectable. The algorithm relaxes the OPTICS steepness definition, allowing for flexible identification of density fault ranges. It can detect overlapping clusters with varying density, hierarchical structures where multiple clusters belong to one cluster, and clustering structures with varying spatial ranges. HST-OPTICS produces clusters with noise exclusion, an undefined total count, clear boundaries, and arbitrary shapes. Simulations have verified its effectiveness in identifying complex hierarchical spatio-temporal clustering structures. Future work could explore practical applications, the design of verification metrics, and efficiency improvements.


## Repository Structure

```plaintext
root
├── simulated_data
│   ├── clustering_structure_1.csv
│   ├── ...
│   └── clustering_structure_7.csv
├── src
│   ├── generate_simulated_data.py
│   ├── clustering_algorithms
│   │   ├── hstoptics.py
│   │   └── stdbscan.py
│   ├── generate_cluster_results.py
│   ├── generate_plots.py
│   ├── parameters.py
│   └── utils.py
├── outputs
│   ├── original_plots
│   │   ├── 2d
│   │   │   ├── clustering_structure_1_2d_plot.png
│   │   │   └── ...
│   │   └── 3d
│   │       └── ...
│   ├── cluster_results
│   │   ├── 2d
│   │   │   ├── clustering_structure_1_2d_cluster_result_hstoptics_param1.csv
│   │   │   ├── clustering_structure_1_2d_cluster_result_stdbscan_param1.csv
│   │   │   ├── clustering_structure_2_2d_cluster_result_hstoptics_param1.csv
│   │   │   ├── clustering_structure_2_2d_cluster_result_stdbscan_param1.csv
│   │   │   └── ...
│   │   └── 3d
│   │       └── ... 
│   ├── cluster_plots
│   │   ├── 2d
│   │   │   ├── clustering_structure_1_2d_cluster_result_hstoptics_param1_plot.png
│   │   │   ├── clustering_structure_1_2d_cluster_result_stdbscan_param1_plot.png
│   │   │   ├── clustering_structure_2_2d_cluster_result_hstoptics_param1_plot.png
│   │   │   ├── clustering_structure_2_2d_cluster_result_stdbscan_param1_plot.png
│   │   │   └── ...
│   │   └── 3d
│   │       └── ...
│   └── reachability_plots
│       ├── 2d
│       │   ├── clustering_structure_2_2d_reachability_plot_param1.png
│       │   └── ...
│       └── 3d
│           └── ...
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```
