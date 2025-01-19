import os
import time
from typing import List, Set, Dict

import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

import seaborn as sns
import matplotlib.pyplot as plt


class HammingDistanceClusterAnalyzer:
    """
    This class demonstrates Method C: Hamming Distance Clustering.

    Steps:
    1. Create vectors (length=100) for each configuration (template, separator, enumerator, choices_order).
       Each vector consists of 0/1 scores, sorted consistently across all configurations.
    2. Calculate Hamming distance between all pairs of configuration vectors.
    3. Apply Hierarchical Clustering based on the distance matrix.
    4. Sample configurations relative to cluster size (or any other desired sampling strategy).
    5. Generate and save a heatmap of the distance matrix for visualization.
    """

    def perform_clustering_for_model(
            self,
            df: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            interesting_datasets: List[str],
            base_results_dir: str
    ) -> None:
        """
        Orchestrates the entire clustering process for each of the specified datasets
        using Hamming distance and hierarchical clustering, and saves a heatmap.

        Args:
            df: DataFrame with raw data (100 rows per configuration).
                Expected columns at least: ['dataset', 'template', 'separator', 'enumerator',
                                           'choices_order', 'score'].
                'score' is assumed to be 0/1 for each sample (row).
            model_name: Name of the model being analyzed (used for directory naming, logs, etc.).
            shots_selected: Number of shots used in the experiment (used for directory naming).
            interesting_datasets: List of dataset names for which we want to run this clustering.
            base_results_dir: Base directory path for saving results (distance matrices, clusters, etc.).
        """
        start_time = time.time()

        # Prepare directory for saving all results
        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_'),
            "HammingClustering"
        )
        os.makedirs(model_dir, exist_ok=True)

        # Iterate over each dataset and run the clustering process
        for dataset in interesting_datasets:
            print(f"Processing dataset: {dataset}")

            # Filter data for current dataset (assuming 'dataset' column in df)
            dataset_df = df[df["dataset"] == dataset].copy()
            if dataset_df.empty:
                print(f"No data for dataset {dataset}. Skipping.")
                continue

            # Create configuration vectors
            config_vectors = self._create_configuration_vectors(dataset_df)

            # If there are fewer than 2 configurations, skip
            if len(config_vectors) < 2:
                print(f"Not enough configurations for dataset {dataset}. Skipping.")
                continue

            # Compute Hamming distance matrix
            config_ids, distance_matrix = self._compute_hamming_distance_matrix(config_vectors)

            # **New step**: Create and save heatmap
            dataset_dir = os.path.join(model_dir, f"dataset_{dataset.replace('/', '_')}")
            os.makedirs(dataset_dir, exist_ok=True)
            heatmap_file = os.path.join(dataset_dir, "hamming_distance_heatmap.png")
            self._plot_and_save_heatmap(distance_matrix, config_ids, heatmap_file)

            # Apply hierarchical clustering
            cluster_assignments = self._hierarchical_clustering(distance_matrix)

            # Sample configurations from each cluster (example strategy)
            sampled_configurations = self._sample_configurations_from_clusters(cluster_assignments, config_ids)

            # Save results (distance matrix, cluster assignments, etc.)
            self._save_clustering_results(
                model_dir=model_dir,
                dataset=dataset,
                config_ids=config_ids,
                distance_matrix=distance_matrix,
                cluster_assignments=cluster_assignments,
                sampled_configurations=sampled_configurations
            )

        total_time = time.time() - start_time
        print(f"Total Hamming Clustering processing time for {model_name}: {total_time:.2f} seconds")
        print("-" * 50)

    def _create_configuration_vectors(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Creates a binary score vector for each unique configuration
        (template, separator, enumerator, choices_order).
        Each configuration in the DataFrame is expected to have 100 rows (score=0 or 1).

        We sort the rows in a consistent manner across configurations (for instance,
        by an index or by a specific ordering of samples), so that all vectors align
        position by position.

        Returns:
            A dictionary mapping a unique configuration string to a NumPy array of shape (100,).
        """
        # We define the unique identifier for each configuration ("quad")
        df["quad"] = (df["template"] + " | " +
                      df["separator"] + " | " +
                      df["enumerator"] + " | " +
                      df["choices_order"])

        # To ensure consistent order, we need a stable sort.
        # If your data doesn't have a separate sample index, rely on the row index.
        df = df.sort_values(by=["quad", df.index.name if df.index.name else df.index])

        # Group by quad and collect the scores into a list (which we turn into an np.ndarray)
        config_vectors = {}
        for quad, group in df.groupby("quad"):
            scores = group["score"].values
            config_vectors[quad] = scores.astype(int)

        return config_vectors

    def _compute_hamming_distance_matrix(self, config_vectors: Dict[str, np.ndarray]):
        """
        Computes the Hamming distance matrix for all pairs of configuration vectors.

        Args:
            config_vectors: Dictionary { config_id: np.array([0,1,...]), ... }

        Returns:
            - A list of configuration ids in the order used for the matrix.
            - A 2D NumPy array (square) of Hamming distances.
        """
        config_ids = sorted(config_vectors.keys())
        vectors = [config_vectors[cid] for cid in config_ids]

        # We use 'pdist' from scipy with 'hamming' metric,
        # Hamming distance is the fraction of positions in which they differ.
        dist_array = pdist(vectors, metric='hamming')
        distance_matrix = squareform(dist_array)

        return config_ids, distance_matrix

    def _plot_and_save_heatmap(self, distance_matrix: np.ndarray, config_ids: List[str], output_file: str) -> None:
        """
        Plots a heatmap of the distance matrix and saves it as a PNG (or other image format).
        The row/column names (config_ids) can be very long, so we arrange the figure
        in a way that is more readable.

        Args:
            distance_matrix: 2D Numpy array of shape (n_configs, n_configs).
            config_ids: List of configuration IDs (str) in the same order as distance_matrix.
            output_file: Path (including filename) where the heatmap will be saved.
        """
        plt.figure(figsize=(12, 10))  # Adjust figsize as needed for readability

        # Create a heatmap using seaborn
        ax = sns.heatmap(
            distance_matrix,
            xticklabels=config_ids,
            yticklabels=config_ids,
            cmap="viridis",
            annot=False  # אפשר לשים True אם רוצים להציג ערכים מספריים
        )

        # כותרת ותוויות
        ax.set_title("Hamming Distance Matrix Heatmap", fontsize=14)
        ax.set_xlabel("Configurations", fontsize=12)
        ax.set_ylabel("Configurations", fontsize=12)

        # לסיבוב הטקסט כך שלא יחפוף
        plt.setp(ax.get_xticklabels(), rotation=70, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to: {output_file}")

    def _hierarchical_clustering(self, distance_matrix: np.ndarray, max_distance: float = 0.2):
        """
        Performs hierarchical clustering on the given distance matrix and returns cluster assignments.

        Args:
            distance_matrix: 2D Numpy array of shape (n_configs, n_configs).
            max_distance: Threshold for cutting the dendrogram into clusters. Adjust to control the number of clusters.

        Returns:
            A 1D array (or list) of cluster IDs, same length as the number of configurations.
        """
        # We need the condensed distance matrix for linkage
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='complete')  # or 'single', 'average', etc.

        cluster_assignments = fcluster(Z, t=max_distance, criterion='distance')
        return cluster_assignments

    def _sample_configurations_from_clusters(self, cluster_assignments, config_ids) -> List[str]:
        """
        Example of how to sample configurations from each cluster, relative to cluster size.

        Args:
            cluster_assignments: 1D array of cluster labels.
            config_ids: List of configuration IDs, in the same order as cluster_assignments.

        Returns:
            A list of sampled configuration IDs.
        """
        df_clusters = pd.DataFrame({
            "config_id": config_ids,
            "cluster": cluster_assignments
        })

        sampled_configurations = []
        for cluster_id, group in df_clusters.groupby("cluster"):
            # Simple rule: pick the first from each cluster
            chosen = group["config_id"].iloc[0]
            sampled_configurations.append(chosen)

        return sampled_configurations

    def _save_clustering_results(
            self,
            model_dir: str,
            dataset: str,
            config_ids: List[str],
            distance_matrix: np.ndarray,
            cluster_assignments: np.ndarray,
            sampled_configurations: List[str]
    ) -> None:
        """
        Saves the distance matrix, cluster assignments, and sampled configurations to disk.

        Args:
            model_dir: Path to the directory where we save all outputs.
            dataset: Dataset name (used in filenames).
            config_ids: Ordered list of configuration IDs corresponding to matrix rows/columns.
            distance_matrix: 2D numpy array of shape (n_configs, n_configs).
            cluster_assignments: 1D array of cluster labels.
            sampled_configurations: A list of sampled configurations from the clustering.
        """
        # Create a sub-directory for this dataset
        dataset_dir = os.path.join(model_dir, f"dataset_{dataset.replace('/', '_')}")
        os.makedirs(dataset_dir, exist_ok=True)

        # Convert distance matrix to DataFrame for easier saving
        dist_df = pd.DataFrame(distance_matrix, index=config_ids, columns=config_ids)
        dist_file = os.path.join(dataset_dir, "hamming_distance_matrix.csv")
        dist_df.to_csv(dist_file)
        print(f"Saved distance matrix to: {dist_file}")

        # Save cluster assignments
        cluster_df = pd.DataFrame({
            "config_id": config_ids,
            "cluster": cluster_assignments
        })
        cluster_file = os.path.join(dataset_dir, "cluster_assignments.csv")
        cluster_df.to_csv(cluster_file, index=False)
        print(f"Saved cluster assignments to: {cluster_file}")

        # Save sampled configurations
        sampled_df = pd.DataFrame({"sampled_configurations": sampled_configurations})
        sampled_file = os.path.join(dataset_dir, "sampled_configurations.csv")
        sampled_df.to_csv(sampled_file, index=False)
        print(f"Saved sampled configurations to: {sampled_file}")
        print("-" * 50)