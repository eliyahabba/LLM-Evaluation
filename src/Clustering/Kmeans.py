import argparse

import pandas as pd
from sklearn.cluster import KMeans
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


class KmeansClustering(Clustering):
    def __init__(self, k, model: str, dataset: str, eval_value: str,
                 random_state: int = ClusteringConstants.RANDOM_STATE,
                 main_results_folder: str = MAIN_RESULTS_PATH):
        super().__init__(model, dataset, eval_value, random_state, main_results_folder)
        self.k = k
        self.kmeans = None

    def fit(self):
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        self.kmeans.fit(self.data)
        self.labels = self.kmeans.labels_
        # Format results as a DataFrame
        results = self.create_results_file(index_name=f"K={self.k}")
        # add the distortions to the results
        results = self.add_distortions(results)
        return results

    def save_labels(self, results: pd.DataFrame) -> None:
        """
        Save the results to the specified path.
        @param results: The results to be saved.
        @return: None
        """
        column_name = f"K={self.k}"
        file_path = self.get_result_output_path("Kmeans")
        self.save_results(results, file_path, column_name)

    def add_distortions(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Add the distortions to the results.
        @param results: The results to which the distortions will be added.
        @return: The results with the distortions added.
        """
        distortions = self.kmeans.inertia_
        # rount distortions to 2 decimal places
        distortions = round(distortions, 2)
        # add new row to the results with the distortions at the template_name column
        row = pd.DataFrame([["Distortions", distortions]], columns=results.columns.values.tolist())
        results = pd.concat([results, row])
        results.reset_index(drop=True, inplace=True)
        return results



class PerformKmeansClustering:
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
            kmeans_clustering = KmeansClustering(k, model, dataset, eval_value=TRAIN_OR_TEST_TYPE)
            kmeans_clustering.load_comparison_matrix()
            results = kmeans_clustering.fit()
            kmeans_clustering.save_labels(results)

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

    perform_kmeans_clustering = PerformKmeansClustering(args.k_min_index, args.k_max_index)
    perform_kmeans_clustering.run_clustering_for_all()
