import numpy as np

import argparse

import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.Analysis.StatisticalTests.CompareSeriesBinaryDataFromTable import CompareSeriesBinaryDataFromTable
from src.utils.Constants import Constants
from src.utils.Utils import Utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
DatasetsConstants = Constants.DatasetsConstants
McNemarTestConstants = Constants.McNemarTestConstants
ResultConstants = Constants.ResultConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
RESULTS_FOLDER = "structured_input"
SHOT = "zero_shot"
FORMAT = "empty_system_format"
TRAIN_OR_TEST_TYPE = "test"

class Clustering:
    def __init__(self, k, model: str, dataset: str, main_results_folder: str = MAIN_RESULTS_PATH):
        self.main_results_folder = main_results_folder
        self.results_folder = f"{main_results_folder}/{RESULTS_FOLDER}/{model}/{dataset}/{SHOT}/{FORMAT}"
        self.k = k
        self.kmeans = KMeans(n_clusters=k)

    def load_results(self, eval_value: str = TRAIN_OR_TEST_TYPE) -> pd.DataFrame:
        """
        Load the results from the specified path.
        @param eval_value:
        @return:
        """
        file_name = f"{ResultConstants.COMPARISON_MATRIX}_{eval_value}_data.csv"
        file_path = f"{self.results_folder}/{file_name}"
        self.results_df = pd.read_csv(file_path)
        self.data = self.results_df.to_numpy()


    def fit(self):
        self.kmeans.fit(self.data)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_
        # Format results as a DataFrame
        results = pd.DataFrame([self.results_df.index,self.labels, self.centroids]).T


    def plot(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, s=50, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=200, alpha=0.5)
        # add a legend
        plt.legend(['Data', 'Centroids'], loc='upper left')
        plt.show()

def cluster_dataset(k, model, dataset):
    clustering = Clustering(k, model, dataset)
    clustering.load_results()
    clustering.fit()
    clustering.plot()

def run_all(k=2):
    for model_key, model_name in tqdm(LLMProcessorConstants.MODEL_NAMES.items()):
        model = Utils.get_model_name(model_name)
        for dataset in tqdm(DatasetsConstants.DATASET_NAMES):
            cluster_dataset(k, model, dataset)
            break

if __name__ == "__main__":
    run_all()
    # exit()
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, choices=LLMProcessorConstants.MODEL_NAMES.keys(),
                      default=list(LLMProcessorConstants.MODEL_NAMES.keys())[1])
    args.add_argument("--dataset", type=str, choices=DatasetsConstants.DATASET_NAMES,
                      default=DatasetsConstants.DATASET_NAMES[0])
    args = args.parse_args()

