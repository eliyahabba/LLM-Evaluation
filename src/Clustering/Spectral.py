import argparse

import pandas as pd
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

from src.Clustering.Clustering import Clustering
from src.utils.Constants import Constants
from src.utils.DatasetsManger import DatasetsManger
from src.utils.Utils import Utils

ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
ClusteringConstants = Constants.ClusteringConstants

MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
TRAIN_OR_TEST_TYPE = "test"


class Spectral(Clustering):
    def __init__(self, k, model: str, dataset: str, eval_value: str,
                 random_state: int = ClusteringConstants.RANDOM_STATE,
                 main_results_folder: str = MAIN_RESULTS_PATH):
        super().__init__(model, dataset, eval_value, random_state, main_results_folder)
        self.k = k
        self.spectral_clustering = None

    def fit(self):
        self.spectral_clustering = SpectralClustering(n_clusters=self.k, affinity='nearest_neighbors',
                                                      random_state=self.random_state)
        self.spectral_clustering.fit(self.data)
        self.labels = self.spectral_clustering.labels_
        # Format results as a DataFrame
        results = self.create_results_file(index_name=f"K={self.k}")
        return results

    def save_labels(self, results: pd.DataFrame) -> None:
        """
        Save the results to the specified path.
        @param results: The results to be saved.
        @return: None
        """
        column_name = f"K={self.k}"
        file_path = self.get_result_output_path("spectral")
        self.save_results(results, file_path, column_name)


class PerformSpectralClustering:
    def __init__(self, k_min_index: int, k_max_index: int) -> None:
        self.k_min_index = k_min_index
        self.k_max_index = k_max_index

    def run_clustering_for_range(self, model: str, dataset: str) -> None:
        """
        Run the clustering for the specified model and dataset.
        @param model: The model to be used for the clustering.
        @param dataset: The dataset to be used for the clustering.
        @return: None
        """
        for k in range(self.k_min_index, self.k_max_index):
            spectral_clustering = Spectral(k, model, dataset, eval_value=TRAIN_OR_TEST_TYPE)
            spectral_clustering.load_comparison_matrix()
            results = spectral_clustering.fit()
            spectral_clustering.save_labels(results)

    def run_clustering_for_all(self) -> None:
        """
        Run the clustering for all the models and datasets.
        @return: None
        """
        for model_key, model_name in tqdm(sorted(LLMProcessorConstants.MODEL_NAMES.items())):
            model = Utils.get_model_name(model_name)
            for dataset in sorted(DatasetsConstants.DATASET_NAMES):
                try:
                    self.run_clustering_for_range(model, dataset)
                except Exception as e:
                    print(f"Error: {e} for model: {model} and dataset: {dataset}")
                    continue


if __name__ == "__main__":
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, choices=LLMProcessorConstants.MODEL_NAMES.keys(),
                      default=list(LLMProcessorConstants.MODEL_NAMES.keys())[1])
    args.add_argument("--dataset", type=str, choices=DatasetsManger.get_dataset_names(),
                      default=DatasetsManger.get_dataset_names()[0])
    args.add_argument("--k_min_index", type=int, default=ClusteringConstants.K_MIN_INDEX,
                      help="The minimum number of clusters.")
    args.add_argument("--k_max_index", type=int, default=ClusteringConstants.K_MAX_INDEX,
                      help="The maximum number of clusters.")
    args = args.parse_args()

    perform_spectral_clustering = PerformSpectralClustering(args.k_min_index, args.k_max_index)
    perform_spectral_clustering.run_clustering_for_all()
