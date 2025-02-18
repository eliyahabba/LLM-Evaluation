import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src.analysis.create_plots.ConfigurationClusterer import ConfigurationClusterer


class HammingDistanceClusterAnalyzerAxes:
    """
    Demonstrates Hamming Distance computation and clustering. Extended to also compute
    a dimension-based distance matrix for each dimension in (template, separator, enumerator, choices_order).
    """

    def perform_clustering_for_model(
            self,
            df: pd.DataFrame,
            model_name: str,
            shots_selected: int,
            dataset: str,
            base_results_dir: str
    ) -> None:
        """
        Orchestrates the entire clustering process for each of the specified datasets
        using Hamming distance. Also computes separate dimension-based distance matrices.
        """
        start_time = time.time()

        # Prepare top-level directory for saving all results
        model_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_')
        )
        os.makedirs(model_dir, exist_ok=True)

        # Iterate over each dataset and run the main process
        print(f"Processing dataset: {dataset}")

        # Filter data for current dataset
        dataset_df = df[df["dataset"] == dataset].copy()
        if dataset_df.empty:
            print(f"No data for dataset {dataset}. Skipping.")
            return

        # Create vectors for full 4-column configurations (the "quad" approach)
        config_vectors = self._create_configuration_vectors(dataset_df)
        if len(config_vectors) < 2:
            print(f"Not enough configurations for dataset {dataset}. Skipping.")
            return

        # Compute and save the "quad" Hamming distance
        config_ids, distance_matrix = self._compute_hamming_distance_matrix(config_vectors)
        dataset_dir = os.path.join(model_dir, f"{dataset.replace('/', '_')}")
        os.makedirs(dataset_dir, exist_ok=True)
        self._plot_and_save_heatmap(distance_matrix, config_ids, dataset_dir)
        clusterer = ConfigurationClusterer()
        results = clusterer.cluster_configs(config_vectors)
        clusterer_file = os.path.join(dataset_dir, "clusterer.npz")
        clusterer.save_compact(results, clusterer_file)

        # Optionally, you can do hierarchical clustering here:
        # cluster_assignments = self._hierarchical_clustering(distance_matrix)
        # sampled_configurations = self._sample_configurations_from_clusters(cluster_assignments, config_ids)
        # self._save_clustering_results(...)

        # -----------------------------
        # New step: Analyze each dimension individually
        # -----------------------------
        # self._analyze_each_dimension(dataset_df, dataset_dir)

        total_time = time.time() - start_time
        print(f"Total Hamming Clustering processing time for {model_name}: {total_time:.2f} seconds")
        print("-" * 50)

    # -------------------------------------------------------------------------
    # 1) The original "quad" approach
    # -------------------------------------------------------------------------
    def _create_configuration_vectors(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Creates a 0/1 vector (length=number_of_rows_per_config) for each unique
        4-column configuration (template, separator, enumerator, choices_order).
        """
        # Build a "quad" identifier
        df["quad"] = (
                df["template"] + " | " +
                df["separator"] + " | " +
                df["enumerator"] + " | " +
                df["choices_order"]
        )

        # Sort to ensure stable ordering
        df = df.sort_values(by=["quad"])

        # Collect scores
        config_vectors = {}
        for quad, group in df.groupby("quad"):
            scores = group["score"].values  # 0/1
            config_vectors[str(quad)] = scores.astype(int)

        return config_vectors

    def _compute_hamming_distance_matrix(self, config_vectors: Dict[str, np.ndarray]):
        """
        Computes pairwise Hamming distances among all configuration vectors.
        Returns (list_of_config_ids, 2D_distance_matrix).
        """
        config_ids = sorted(config_vectors.keys())
        vectors = [config_vectors[cid] for cid in config_ids]

        # Hamming distance = fraction of positions that differ
        dist_array = pdist(vectors, metric='hamming')
        distance_matrix = squareform(dist_array)

        return config_ids, distance_matrix

    def _plot_and_save_heatmap(self, distance_matrix: np.ndarray, config_ids: List[str], dataset_dir: str):
        """
        Plot and save a heatmap of the given distance_matrix. Also save it as a parquet file.
        """
        # plt.figure(figsize=(12, 10))
        # ax = sns.heatmap(
        #     distance_matrix,
        #     xticklabels=config_ids,
        #     yticklabels=config_ids,
        #     cmap="viridis",
        #     annot=False
        # )
        # ax.set_title("Hamming Distance Matrix Heatmap", fontsize=14)
        # ax.set_xlabel("Configurations", fontsize=12)
        # ax.set_ylabel("Configurations", fontsize=12)
        #
        # plt.setp(ax.get_xticklabels(), rotation=70, ha="right", rotation_mode="anchor")
        # plt.setp(ax.get_yticklabels(), rotation=0)

        # heatmap_file = os.path.join(dataset_dir, "hamming_distance_heatmap.png")
        # plt.tight_layout()
        # plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved heatmap to: {heatmap_file}")

        # Save matrix to parquet
        data_output_path = os.path.join(dataset_dir, "hamming_distance_matrix.parquet")
        dist_df = pd.DataFrame(distance_matrix, index=config_ids, columns=config_ids)
        dist_df.to_parquet(data_output_path)
        # save also the config_ids dict
        config_ids_df = pd.DataFrame(config_ids, columns=["config_id"])
        config_ids_df.to_parquet(os.path.join(dataset_dir, "config_ids.parquet"))
        print(f"Saved distance matrix to: {data_output_path}")

    # -------------------------------------------------------------------------
    # 2) Dimension-based approach
    # -------------------------------------------------------------------------
    def _create_majority_vectors_for_dimension(self, df: pd.DataFrame, focus_col: str) -> Dict[str, np.ndarray]:
        """
        Create a single 0/1, length=100 vector for each value of 'focus_col' by taking a
        majority vote across all configurations that share that focus_col value.

        For example, if focus_col='template' and dimension value='T1', we gather all
        configurations whose template='T1', each of which has 100 rows (score=0/1).
        For each sample_idx in 0..99, we see how many of those configurations had a '1'
        vs. '0' at that index. Whichever is higher is the chosen bit in the final vector.

        Requirements:
          - The DataFrame must have 'sample_idx' from 0..99 for each configuration row.
        """
        # We assume each row in df has: focus_col, sample_idx, score, plus other columns.
        # We'll group by (focus_col, sample_idx), compute the fraction of 1's, then do majority rule.
        grouped = (df
                   .groupby([focus_col, "sample_index"])["score"]
                   .mean()  # average of 0/1 in that group
                   .reset_index(name="mean_score"))

        # 'grouped' now has rows: (dim_value, sample_idx, mean_score).
        # We pivot it so that rows=sample_idx (0..99), columns=dim_value, values=mean_score.
        pivot_df = grouped.pivot(index="sample_index", columns=focus_col, values="mean_score")

        # Now pivot_df is a 100-row DataFrame (index=sample_idx), columns = dimension values
        # Each cell is the average of 0/1 for that (dim_value, sample_idx). We do majority = (mean_score > 0.5).
        dim_vectors = {}
        for dim_val in pivot_df.columns:
            mean_scores_col = pivot_df[dim_val]  # if no data, fill with 0
            # Convert to majority 0/1
            majority_array = (mean_scores_col > 0.5).astype(int).values
            # This should be length=100 if sample_idx goes 0..99
            dim_vectors[str(dim_val)] = majority_array

        return dim_vectors

    def _analyze_each_dimension(self, dataset_df: pd.DataFrame, dataset_dir: str) -> None:
        """
        For each of the four columns, build a 100-length majority-vote vector for each
        dimension value, then compute a distance matrix among those vectors.
        """
        dimensions = ["template", "separator", "enumerator", "choices_order"]

        for dim in dimensions:
            print(f"[Dimension: {dim}] Computing majority-vote vectors...")
            dim_subdir = os.path.join(dataset_dir, dim)
            os.makedirs(dim_subdir, exist_ok=True)

            # Build dimension-based vectors (one vector per dimension value)
            dim_vectors = self._create_majority_vectors_for_dimension(dataset_df, dim)
            if len(dim_vectors) < 2:
                print(f"[Dimension: {dim}] Not enough values to compute distance.")
                continue

            # Compute the distance matrix
            dim_ids, dist_mat = self._compute_hamming_distance_matrix(dim_vectors)

            # Plot and save
            self._plot_and_save_dimension_heatmap(dist_mat, dim_ids, dim_subdir, dim_name=dim)

    def _create_dimension_vectors_for_dim(self, df: pd.DataFrame, focus_col: str) -> Dict[str, np.ndarray]:
        """
        Build a single 0/1 vector for each unique value of 'focus_col'.
        In this naive example, we simply take all rows that have a certain focus_col value
        and concatenate their 0/1 'score' values in sorted order.

        If some dimension values have more rows than others, you will end up with vectors
        of differing lengths, which will break Hamming distance. In a real scenario, you'd
        define how to unify or aggregate them so that every dimension value has the same
        vector length (e.g., by average, by majority vote, by a consistent indexing scheme, etc.).
        """
        df_copy = df.copy()

        # We'll identify each dimension value by its raw text
        df_copy = df_copy.sort_values(by=[focus_col])

        dim_vectors = {}
        for dim_val, group in df_copy.groupby(focus_col):
            scores = group["score"].values  # potential mismatch in lengths across dimension values
            dim_vectors[str(dim_val)] = scores.astype(int)

        return dim_vectors

    def _plot_and_save_dimension_heatmap(
            self,
            distance_matrix: np.ndarray,
            dim_ids: List[str],
            dim_subdir: str,
            dim_name: str
    ) -> None:
        """
        Plot and save a distance heatmap specifically for one dimension.
        Also save it as a parquet file.
        """
        # plt.figure(figsize=(12, 10))
        # ax = sns.heatmap(
        #     distance_matrix,
        #     xticklabels=dim_ids,
        #     yticklabels=dim_ids,
        #     cmap="viridis",
        #     annot=False
        # )
        # ax.set_title(f"Hamming Distance Matrix - Dimension: {dim_name}", fontsize=14)
        # ax.set_xlabel(f"{dim_name} Values", fontsize=12)
        # ax.set_ylabel(f"{dim_name} Values", fontsize=12)
        #
        # plt.setp(ax.get_xticklabels(), rotation=70, ha="right", rotation_mode="anchor")
        # plt.setp(ax.get_yticklabels(), rotation=0)

        # Save figure
        # heatmap_file = os.path.join(dim_subdir, f"hamming_distance_heatmap.png")
        # plt.tight_layout()
        # plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved {dim_name} dimension heatmap to: {heatmap_file}")

        # Save matrix to parquet
        data_output_path = os.path.join(dim_subdir, f"hamming_distance_matrix.parquet")
        dist_df = pd.DataFrame(distance_matrix, index=dim_ids, columns=dim_ids)
        dist_df.to_parquet(data_output_path)
        print(f"Saved {dim_name} dimension distance matrix to: {data_output_path}")

    # -------------------------------------------------------------------------
    # (Optional) Hierarchical clustering and results saving
    # -------------------------------------------------------------------------
    def _hierarchical_clustering(self, distance_matrix: np.ndarray, max_distance: float = 0.2):
        """
        Performs hierarchical clustering on the given distance matrix
        and returns cluster assignments.
        """
        from scipy.cluster.hierarchy import linkage, fcluster

        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='complete')
        cluster_assignments = fcluster(Z, t=max_distance, criterion='distance')
        return cluster_assignments

    def _sample_configurations_from_clusters(self, cluster_assignments, config_ids) -> List[str]:
        """
        Example of how to sample configurations from each cluster, relative to cluster size.
        """
        df_clusters = pd.DataFrame({
            "config_id": config_ids,
            "cluster": cluster_assignments
        })

        sampled_configurations = []
        for cluster_id, group in df_clusters.groupby("cluster"):
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
        """
        dataset_dir = os.path.join(model_dir, f"{dataset.replace('/', '_')}")
        os.makedirs(dataset_dir, exist_ok=True)

        dist_df = pd.DataFrame(distance_matrix, index=config_ids, columns=config_ids)
        dist_file = os.path.join(dataset_dir, "hamming_distance_matrix.csv")
        dist_df.to_csv(dist_file)
        print(f"Saved distance matrix to: {dist_file}")

        cluster_df = pd.DataFrame({
            "config_id": config_ids,
            "cluster": cluster_assignments
        })
        cluster_file = os.path.join(dataset_dir, "cluster_assignments.csv")
        cluster_df.to_csv(cluster_file, index=False)
        print(f"Saved cluster assignments to: {cluster_file}")

        sampled_df = pd.DataFrame({"sampled_configurations": sampled_configurations})
        sampled_file = os.path.join(dataset_dir, "sampled_configurations.csv")
        sampled_df.to_csv(sampled_file, index=False)
        print(f"Saved sampled configurations to: {sampled_file}")
        print("-" * 50)
