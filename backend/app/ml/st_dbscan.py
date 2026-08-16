import sys
from collections import deque
import numpy as np
import pandas as pd
from typing import Optional
from backend.app.core.logger import logging
from backend.app.core.exceptions import CustomException
from sklearn.metrics.pairwise import haversine_distances

class STDBSCAN:
    """
    Fits ST-DBSCAN and predicts cluster labels for a spatio-temporal dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing spatial coordinates and timestamps.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    time_col : str
        Timestamp or datetime column name.
        
    Returns:
    --------
    np.ndarray:
        Array of cluster labels (-1 for noise/isolated points, >=0 for cluster IDs).
    """
    def __init__(self, eps1: float = 10.0, eps2: float = 7.0, min_samples: int = 5):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.labels_: Optional[np.ndarray] = None

    @staticmethod
    def _haversine_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        # 1. Convert lat/lon degrees to radians in shape (N, 2)
        coords_in_radians = np.radians(np.column_stack([lats, lons]))
        
        # 2. Compute pairwise distance matrix in radians
        result_in_radians = haversine_distances(coords_in_radians)
        
        # 3. Multiply by Earth radius (6371.0088 km)
        earth_radius_km = 6371.0088
        return result_in_radians * earth_radius_km

    def fit_predict(
        self, 
        df: pd.DataFrame, 
        lat_col: str = 'latitude', 
        lon_col: str = 'longitude', 
        time_col: str = 'timestamp'
    ) -> np.ndarray:
        try:
            logging.info(f"Executing BFS-based ST-DBSCAN on {len(df)} records.")
            logging.info(f"Hyperparameters: eps1 (spatial) = {self.eps1} km, eps2 (temporal) = {self.eps2} days, min_samples = {self.min_samples}")
            n_points = len(df)
            if n_points == 0:
                logging.warning("Received empty DataFrame for ST-DBSCAN.")
                return np.array([])

            # Extract spatial coordinates
            lats = df[lat_col].values
            lons = df[lon_col].values
            
            # Convert timestamp column to days relative to epoch
            timestamps = pd.to_datetime(df[time_col])
            time_in_days = (timestamps - pd.Timestamp("1970-01-01")).dt.total_seconds().values / 86400.0

            # 1. Distance Matrices
            spatial_dist = self._haversine_distance_matrix(lats, lons)
            temporal_dist = np.abs(time_in_days[:, np.newaxis] - time_in_days[np.newaxis, :])

            # 2. Boolean Adjacency Matrix
            st_neighbors = (spatial_dist <= self.eps1) & (temporal_dist <= self.eps2)

            # 3. BFS-based Cluster Expansion
            labels = np.full(n_points, -1, dtype=int)
            visited = np.zeros(n_points, dtype=bool)
            cluster_id = 0

            for i in range(n_points):
                if visited[i]:
                    continue

                neighbors = np.where(st_neighbors[i])[0]

                # Check if core point
                if len(neighbors) < self.min_samples:
                    labels[i] = -1  # Mark as noise
                    visited[i] = True
                else:
                    # Start a new cluster using BFS
                    labels[i] = cluster_id
                    visited[i] = True
                    
                    # Initialize BFS Queue with core point neighbors
                    queue = deque()
                    in_queue = set()

                    for nbr in neighbors:
                        if nbr != i:
                            if labels[nbr] == -1:
                                labels[nbr] = cluster_id
                            if not visited[nbr]:
                                visited[nbr] = True
                                queue.append(nbr)
                                in_queue.add(nbr)

                    while queue:
                        curr = queue.popleft()

                        # Expand cluster if 'curr' is also a core point
                        if not visited[curr]:
                            visited[curr] = True
                            curr_neighbors = np.where(st_neighbors[curr])[0]

                            if len(curr_neighbors) >= self.min_samples:
                                for nbr in curr_neighbors:
                                    if nbr not in in_queue and not visited[nbr]:
                                        queue.append(nbr)
                                        in_queue.add(nbr)

                        # Assign cluster ID if point is unassigned or previously marked noise
                        if labels[curr] == -1:
                            labels[curr] = cluster_id

                    cluster_id += 1

            self.labels_ = labels
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            
            logging.info(f"BFS ST-DBSCAN completed: {n_clusters} clusters, {n_noise} noise points.")
            return labels

        except Exception as e:
            logging.error("Unhandled error in BFS ST-DBSCAN.")
            raise CustomException(e, sys)