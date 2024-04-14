import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.utils.Constants import Constants
from src.utils.Utils import Utils

ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
DatasetsConstants = Constants.DatasetsConstants
ResultConstants = Constants.ResultConstants
ClusteringConstants = Constants.ClusteringConstants

MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
RESULTS_FOLDER = "structured_input"
SHOT = "zero_shot"
FORMAT = "empty_system_format"
TRAIN_OR_TEST_TYPE = "test"


class KmeansClustering:
    def __init__(self, k, model: str, dataset: str, eval_value: str,
                 main_results_folder: str = MAIN_RESULTS_PATH):
        self.centroids = None
        self.labels = None
        self.data = None
        self.results_df = None
        self.main_results_folder = main_results_folder
        self.results_folder = f"{main_results_folder}/{RESULTS_FOLDER}/{model}/{dataset}/{SHOT}/{FORMAT}"
        self.k = k
        self.eval_value = eval_value
        self.kmeans = KMeans(n_clusters=k)

    def load_results(self) -> None:
        """
        Load the results from the specified path.
        @param eval_value:
        @return:
        """
        file_name = f"{ResultConstants.COMPARISON_MATRIX}_{self.eval_value}_data.csv"
        file_path = f"{self.results_folder}/{file_name}"
        self.results_df = pd.read_csv(file_path)
        self.data = self.results_df.to_numpy()
        # transpose the data, so that the templates are the rows and the features are the columns
        self.data = self.data.T

    def fit(self):
        self.kmeans.fit(self.data)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_
        # Format results as a DataFrame
        columns = list(map(lambda x: x.split("experiment_")[1], self.results_df.columns.values))
        results = pd.DataFrame([self.labels],
                               columns=columns, index=[f"K={self.k}"]).T
        results.index.name = "template_name"
        results.reset_index(inplace=True, drop=False)
        return results

    def save_results(self, results):
        file_path = Path(f"{self.results_folder}/{ResultConstants.CLUSTERING_RESULTS}_{self.eval_value}_data.csv")
        # if the file already exists, read the file, and append the new results (if they don't already exist,
        # if they do, update them)
        if file_path.exists():
            existing_results = pd.read_csv(file_path, index_col=0)
            # check if there is column with the same name as the new results Cluster column
            if f"K={self.k}" in existing_results.columns:
                # update the results
                existing_results[f"K={self.k}"] = results[f"K={self.k}"]
                updated_results = existing_results
            else:
                # concat on all the columns
                updated_results = pd.merge(existing_results, results, on='template_name', how='inner')
                updated_results.to_csv(file_path)
        else:
            results.to_csv(file_path)


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
            clustering = KmeansClustering(k, model, dataset, eval_value=TRAIN_OR_TEST_TYPE)
            clustering.load_results()
            results = clustering.fit()
            clustering.save_results(results)

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
    args.add_argument("--dataset", type=str, choices=DatasetsConstants.DATASET_NAMES,
                      default=DatasetsConstants.DATASET_NAMES[0])
    args.add_argument("--k_min_index", type=int, default=ClusteringConstants.K_MIN_INDEX,
                      help="The minimum number of clusters.")
    args.add_argument("--k_max_index", type=int, default=ClusteringConstants.K_MAX_INDEX,
                      help="The maximum number of clusters.")
    args = args.parse_args()

    clustering = PerformKmeansClustering(args.k_min_index, args.k_max_index)
    clustering.run_clustering_for_all()
