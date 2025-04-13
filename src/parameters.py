# parameters.py

param_sets = {
    "hstoptics": {
        "clustering_structure_1": {
            2: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.4,  "window_size": 500},
            3: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.4,  "window_size": 500},
        },
        "clustering_structure_2": {
            2: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.4,  "window_size": 500},
            3: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.4,  "window_size": 500},
        },
        "clustering_structure_3": {
            2: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.4,  "window_size": 500},
            3: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.4,  "window_size": 500},
        },
        "clustering_structure_4": {
            2: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.34, "window_size": 500},
            3: {"eps_s": 1.2, "eps_t": 1.2, "w_s": 1, "w_t": 1, "min_pts": 50,  "diff": 0.26, "window_size": 300},
        },
        "clustering_structure_5": {
            2: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.45, "window_size": 500},
            3: {"eps_s": 1.2, "eps_t": 1.2, "w_s": 1, "w_t": 1, "min_pts": 50,  "diff": 0.37, "window_size": 500},
        },
        "clustering_structure_6": {
            2: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.17, "window_size": 570},
            3: {"eps_s": 1.2, "eps_t": 1.2, "w_s": 1, "w_t": 1, "min_pts": 50,  "diff": 0.19, "window_size": 650},
        },
        "clustering_structure_7": {
            2: {"eps_s": 1.5, "eps_t": 1.5, "w_s": 1, "w_t": 1, "min_pts": 100, "diff": 0.25, "window_size": 650},
            3: {"eps_s": 1.2, "eps_t": 1.2, "w_s": 1, "w_t": 1, "min_pts": 50,  "diff": 0.21, "window_size": 250},
        }
    },
    "stdbscan": {
        "clustering_structure_1": {
            2: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
            3: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
        },
        "clustering_structure_2": {
            2: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
            3: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
        },
        "clustering_structure_3": {
            2: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
            3: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
        },
        "clustering_structure_4": {
            2: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
            3: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
        },
        "clustering_structure_5": {
            2: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
            3: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
        },
        "clustering_structure_6": {
            2: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
            3: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
        },
        "clustering_structure_7": {
            2: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
            3: {"spatial_threshold": 1, "temporal_threshold": 1, "min_neighbors": 6},
        }
    }
}
