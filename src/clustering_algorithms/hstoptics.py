# hstoptics.py
"""
HST-OPTICS: The proposed spatio-temporal clustering algorithm.
"""
import re
import numpy as np
import pandas as pd
from typing import List
from src.utils import get_distance, get_st_distance


class HSTOPTICS:
    """
    Hierarchical Spatio-Temporal Ordering Points To Identify Clustering Structure (HST-OPTICS) algorithm.

    Args:
        eps_s: Radius of the spatial epsilon neighborhood.
        eps_t: Radius of the temporal epsilon neighborhood.
        min_pts: Number of minimum points required in the eps_s and eps_t
                 neighborhood for core points (including the point itself).
        w_s: Weight of spatial distance.
        w_t: Weight of temporal distance.
        diff: Threshold for detecting a significant change (fault) in reachability score.
        window_size: Number of following points to consider for slope calculation.

    Attributes:
        data_df:
            Input DataFrame containing the data to be clustered with id in first column.
        cluster_profile:
            A DataFrame storing the hierarchical spatio-temporal clustering profile. It includes:
            - ordered_id: ID of points sorted by visiting order.
            - reach_score: The computed reachability score of each point.
            - fault: Indicates whether a point is a fault point (1 or -1) or not (0).
            - level: The hierarchical level to which the point belongs, defined between fault points.
        cluster_result:
            A DataFrame containing the original input data merged with clustering result.
            Includes all original columns and an additional 'cluster' column denoting
            the hierarchical cluster assignment.
    """

    def __init__(
        self,
        eps_s: float,
        eps_t: float,
        min_pts: int
    ):
        """
        Initialize the HST-OPTICS clustering instance.
        """
        self.eps_s = eps_s
        self.eps_t = eps_t
        self.min_pts = min_pts
        self.w_s = None
        self.w_t = None
        self.diff = None
        self.window_size = None
        self.cluster_profile = None
        self.cluster_results = None

    def retrieve_neighbors(
        self,
        one_point: pd.Series,
        all_points: pd.DataFrame,
        xyt_colnames: List[str] = ['x', 'y', 't'],
    ) -> pd.DataFrame:
        """
        Retrieves spatio-temporal neighbors of a point within eps_s and eps_t.

        Args:
            one_point: The reference point.
            all_points: The DataFrame of other points.
            xyt_colnames: List of column names for ['x', 'y', 't'].

        Returns:
            A DataFrame of neighboring points.
        """

        x_col, y_col, t_col = xyt_colnames[0], xyt_colnames[1], xyt_colnames[2]

        # Find temporal neighbors: points within eps_t time range
        t_neighbors = all_points[
            (all_points[t_col] >= one_point[t_col] - self.eps_t) &
            (all_points[t_col] <= one_point[t_col] + self.eps_t)
        ]

        # Find spatial neighbors within eps_s distance
        s_dists = get_distance(one_point, t_neighbors, [x_col, y_col])
        st_neighbors = t_neighbors[s_dists <= self.eps_s].reset_index(drop=True).copy()

        return st_neighbors

    def compute_order_and_reachability(
        self,
        data_df: pd.DataFrame,
        xyt_colnames: List[str] = ['x', 'y', 't'],
        w_s: float = 1.0,
        w_t: float = 1.0
    ) -> pd.DataFrame:
        """
        Orders spatio-temporal points and computes their reachability distances.

        Args:
            data_df: A DataFrame containing spatio-temporal points.
            xyt_colnames: List of column names for ['x', 'y', 't'].

        Returns:
            reachability_scores: A DataFrame containing order and reachability scores of points.
        """
        self.w_s = w_s
        self.w_t = w_t

        # Copy input DataFrame and assign a unique ID to each point
        data_df_copy = data_df.copy().reset_index(drop=True)
        data_df_copy['id'] = data_df_copy.index
        self.data_df = data_df_copy

        ordered_id = []        # Stores order of visited points
        reach_score = []       # Stores reachability distances
        points_to_be_visited = pd.DataFrame(
            columns=data_df_copy.columns.to_list() + ['reach_dist']
        )  # Queue of candidate points

        for _ in range(len(data_df_copy)):
            # Visit each point one by one

            if points_to_be_visited.empty:
                # No points in the queue: pick first unvisited point
                visited_id = sorted(list(set(data_df_copy['id']) - set(ordered_id)))[0]
                reach_score.append(np.inf)
            else:
                # Pick the point with the smallest reachability distance
                visited_id = points_to_be_visited.loc[
                    points_to_be_visited['reach_dist'].idxmin(), 'id'
                ]
                reach_score.append(
                    points_to_be_visited['reach_dist'].min()
                )

            # Record visited point
            ordered_id.append(visited_id)
            visited_point = data_df_copy.loc[visited_id]

            # Retrieve spatio-temporal neighbors of the visited point
            st_neighbors = self.retrieve_neighbors(visited_point, data_df_copy, xyt_colnames)

            # If the point is a core point
            if len(st_neighbors) >= self.min_pts:
                # Compute spatio-temporal distances to neighbors
                st_neighbors['st_dist'] = get_st_distance(
                    visited_point, st_neighbors, self.w_s, self.w_t, xyt_colnames
                )

                # Core distance is distance to the min_pts-th closest neighbor
                core_dist = np.sort(st_neighbors['st_dist'].values)[self.min_pts - 1]

                # Reachability distance = max(core_dist, actual st-distance)
                st_neighbors = st_neighbors.rename(columns={'st_dist': 'reach_dist'})
                st_neighbors['reach_dist'] = st_neighbors['reach_dist'].apply(
                    lambda st_dist: max(st_dist, core_dist)
                )

                # Update points to be visited
                dfs = [df for df in [points_to_be_visited, st_neighbors] if not df.empty]
                if dfs:
                    points_to_be_visited = pd.concat(dfs, ignore_index=True)

                    # Keep the closest (smallest reach_dist) point for each id
                    points_to_be_visited = points_to_be_visited.sort_values(
                        by='reach_dist', kind='stable'
                    ).drop_duplicates(subset='id', keep='first')

            # Remove the just-visited point from the queue
            points_to_be_visited = points_to_be_visited[
                ~points_to_be_visited['id'].isin(ordered_id)
            ].reset_index(drop=True)

        # Construct cluster profile
        reachability_scores = pd.DataFrame({
            'ordered_id': ordered_id,
            'reach_score': reach_score
        })

        self.cluster_profile = reachability_scores
        return reachability_scores

    def validate_reachability_scores(self) -> bool:
        """
        Validate if all reachability scores are either Inf or -Inf.

        Returns:
            bool: True if all values are Inf or -Inf, False otherwise.
        """
        reach_scores = self.cluster_profile['reach_score']
        return np.all(np.isin(reach_scores, [np.inf, -np.inf]))

    def find_faults(self, diff: float, window_size: int) -> pd.Series:
        """
        Identifies significant changes (faults) in the reachability plot using slope-based window scanning.

        Return:
            A Series with with the same length as input,
            marking 1 for upward faults, -1 for downward faults, and 0 for no faults.
        """
        self.diff = diff
        self.window_size = window_size

        n = self.cluster_profile.shape[0]

        # Initailize
        faults = np.zeros(n, dtype=int)
        smallest_slope = np.inf
        largest_slope = -np.inf
        best_start_order = None
        best_end_order = None
        current_up_down = 0
        nearest_order = None
        current_slope = 0
        slope = 0

        for order in range(n):
            # Set up searching window
            window_start = order + 1
            window_end = min(n, window_start + window_size)

            # Find nearest order that fits condition
            diffs = self.cluster_profile['reach_score'].iloc[window_start:window_end] - self.cluster_profile['reach_score'].iloc[order]
            conditions = (diffs < -self.diff) | (diffs > self.diff)
            nearest_order_in_window = next((i for i, condition in enumerate(conditions) if condition), None)
            nearest_order = window_start + nearest_order_in_window if nearest_order_in_window is not None else None

            if nearest_order is None:
                current_slope = 0
            else:
                # Update largest_start_order or smallest_end_order
                current_slope = diffs.iloc[nearest_order_in_window] / (nearest_order - order)

            if (nearest_order is None) or (current_slope > 0 and slope < 0) or (current_slope < 0 and slope > 0):
                # Record previous faults
                if current_up_down == 1 and best_start_order is not None:
                    # If upward trend doesn't continue, record upward fault position
                    faults[best_start_order] = current_up_down
                    best_start_order = None
                    largest_slope = -np.inf
                elif current_up_down == -1 and best_end_order is not None:
                    # If downward trend doesn't continue, record downward fault position
                    faults[best_end_order] = current_up_down
                    best_end_order = None
                    smallest_slope = np.inf

                if nearest_order is None:
                    # If there is no order fitting condition, move to next order
                    current_up_down = 0
                    continue

            # Renew slope
            slope = current_slope

            if slope > 0:
                current_up_down = 1
                if slope >= largest_slope:
                    largest_slope = slope
                    best_start_order = order
            elif slope < 0:
                current_up_down = -1
                if slope <= smallest_slope:
                    smallest_slope = slope
                    best_end_order = nearest_order

        self.cluster_profile['fault'] = faults
        return faults

    def assign_density_levels(self) -> pd.Series:
        """
        Assigns hierarchical density levels based on fault directions in the reachability plot.

        Returns:
            A pandas Series of levels (int), aligned with the cluster profile index.
        """
        faults = self.cluster_profile['fault']
        n = self.cluster_profile.shape[0]

        levels = [0] * n
        current_level = 0

        for i in range(1, n):
            current_level += faults[i]

            # Record level for current point
            levels[i] = current_level

        levels = [lvl - max(levels) for lvl in levels]

        # Set level to 0 when reach_score is inf
        levels = [
            0 if np.isinf(self.cluster_profile['reach_score'][i]) else levels[i]
            for i in range(n)
        ]

        # Set last level to 0
        levels[-1] = 0

        self.cluster_profile['level'] = levels
        return levels

    def assign_clusters(self):
        """
        Assigns cluster labels to each point based on the hierarchical density levels.

        Return:
            A DataFrame that joins original data (`data_df`) with cluster assignments.
        """
        levels = self.cluster_profile['level'].to_list()
        ordered_ids = self.cluster_profile['ordered_id'].to_list()
        n = self.cluster_profile.shape[0]
        deepest_level = min(levels)

        # Initialize point-cluster membership table
        point_cluster_membership = pd.DataFrame({
            'ordered_id': ordered_ids,
            'cluster': [-1] * n
        })

        # for each level
        n_clusters = 0
        if deepest_level != 0:
            for jj in range(-1, deepest_level - 1, -1):
                break_indices = [i for i, lvl in enumerate(levels) if lvl > jj]

                if break_indices:
                    # Create interval bins between break points
                    bins = pd.cut(range(n), bins=[-1] + break_indices + [n], right=False)
                    bin_counts = bins.value_counts().sort_index()

                    cluster_positions = []
                    for interval_str, count in bin_counts.items():
                        if count > 1:
                            start, end = map(int, re.findall(r'\d+', str(interval_str)))
                            cluster_positions.append((start + 1, end - 1))

                    for start, end in cluster_positions:
                        in_cluster_ii = [(start <= i <= end) for i in range(n)]
                        cluster_col = [1 if val else 0 for val in in_cluster_ii]
                        n_clusters += 1
                        col_name = f'cluster{n_clusters}'

                        point_cluster_membership[col_name] = cluster_col
                        for i, val in enumerate(in_cluster_ii):
                            if val:
                                point_cluster_membership.at[i, 'cluster'] = n_clusters

        # Merge with original data by matching ordered_id and id
        cluster_result = self.data_df.copy()
        cluster_result = cluster_result.merge(
            point_cluster_membership, left_on='id', right_on='ordered_id', how='left'
        )
        cluster_result = cluster_result.drop(columns=['id'])

        self.cluster_result = cluster_result
        return cluster_result
