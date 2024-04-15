import argparse

import hdbscan
import pandas as pd
from tqdm import tqdm

from src.Clustering.Clustering import Clustering
from src.utils.Constants import Constants
from src.utils.Utils import Utils

ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
DatasetsConstants = Constants.DatasetsConstants
ClusteringConstants = Constants.ClusteringConstants

MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
TRAIN_OR_TEST_TYPE = "test"


class Hdbscan(Clustering):
    def __init__(self, model: str, dataset: str, eval_value: str,
                 main_results_folder: str = MAIN_RESULTS_PATH,
                 min_cluster_size=5, min_samples=None):
        super().__init__(model=model, dataset=dataset, eval_value=eval_value, main_results_folder=main_results_folder)

        self.clusterer = None
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit(self) -> pd.DataFrame:
        """
        Fit the clustering model.
        @return:
        """
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        self.labels = self.clusterer.fit_predict(self.data)
        results = self.create_results_file(index_name=f"min_cluster_size={self.min_cluster_size}")
        return results

    def save_labels(self, results: pd.DataFrame) -> None:
        """
        Save the results to the specified path.
        @param results: The results to be saved.
        @return: None
        """
        column_name = f"min_cluster_size={self.min_cluster_size}"
        file_path = self.get_result_output_path("Hdbscan")
        self.save_results(results, file_path, column_name)


class PerformHdbscanClustering:
    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def run_hdbscan_clustering(self, model, dataset):
        hdbscan_clustering = Hdbscan(model, dataset, eval_value=TRAIN_OR_TEST_TYPE,
                                     min_cluster_size=self.min_cluster_size,
                                     min_samples=self.min_samples)
        hdbscan_clustering.load_comparison_matrix()
        results = hdbscan_clustering.fit()
        hdbscan_clustering.save_labels(results)

    def run_clustering_for_all(self):
        for model_key, model_name in tqdm(sorted(LLMProcessorConstants.MODEL_NAMES.items())):
            model = Utils.get_model_name(model_name)
            for dataset in sorted(DatasetsConstants.DATASET_NAMES):
                try:
                    self.run_hdbscan_clustering(model, dataset)
                except Exception as e:
                    print(f"Error: {e} for model: {model} and dataset: {dataset}")
                    continue


if __name__ == "__main__":
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, choices=LLMProcessorConstants.MODEL_NAMES.keys(),
                      default=list(LLMProcessorConstants.MODEL_NAMES.keys())[1])
    args.add_argument("--dataset", type=str, choices=DatasetsConstants.DATASET_NAMES,
                      default=DatasetsConstants.DATASET_NAMES[0])
    args.add_argument("--min_cluster_size", type=int, default=ClusteringConstants.MIN_CLUSTER_SIZE,
                      help="The minimum number of samples in a cluster.")
    args.add_argument("--min_samples", type=int, default=ClusteringConstants.MIN_SAMPLES,
                      help="The number of samples in a neighborhood for a point to be considered as a core point.")
    args = args.parse_args()

    perform_hdbscan_clustering = PerformHdbscanClustering(min_cluster_size=args.min_cluster_size,
                                                          min_samples=args.min_samples)
    perform_hdbscan_clustering.run_clustering_for_all()
