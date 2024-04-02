import argparse
import itertools
import json

import numpy as np
import pandas as pd

from src.utils.Constants import Constants
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import cochrans_q

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
DatasetsConstants = Constants.DatasetsConstants

MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
RESULTS_FOLDER = "structured_input"
SHOT = "zero_shot"
FORMAT = "empty_system_format"
TRAIN_OR_TEST_TYPE = "test"

ALPHA = 0.05

class McNemarTest:
    """
    Class to perform McNemar test on the results that are obtained from different models and datasets.
    """

    def __init__(self, model: str, dataset: str,
                 main_results_folder: str = MAIN_RESULTS_PATH):
        self.main_results_folder = main_results_folder
        self.results_folder = f"{main_results_folder}/{RESULTS_FOLDER}/{model}/{dataset}/{SHOT}/{FORMAT}"

    def load_results(self) -> pd.DataFrame:
        """
        Load the results from the specified path.
        @return: The results
        """
        f"{TRAIN_OR_TEST_TYPE}_accuracy_results.csv"
        results = pd.read_csv(f"{self.results_folder}/{TRAIN_OR_TEST_TYPE}_accuracy_results.csv")
        return results

    def perform_cochrans_q_test(self, results):
        """
        Perform Cochran's Q test on the given results.
        """
        # Perform Cochran's Q test
        result = cochrans_q(results)
        # Print results
        print("Cochran's Q Test Statistic:", result.statistic)
        print(f"Cochran's Q Test P-value:{result.pvalue}")

    def perform_mcnemar_test(self, results):
        """
        Perform McNemar test on the given results.
        @param results1: Results from model 1
        @param results2: Results from model 2
        """
        # take all the pairs of columns and for each pair, create a contingency table
        test_results = []
        column_pairs = itertools.combinations(results.columns, 2)

        df = pd.DataFrame(columns=results.columns)
        # add rows index with the same index as the columns
        for col in results.columns:
            df.loc[col] = np.nan
        df.set_index(results.columns, inplace=True)
        # create this to be the index of the dataframe
        for column1, column2 in column_pairs:
            contingency_table = pd.crosstab(results[column1], results[column2])
            # Perform McNemar test
            result = mcnemar(contingency_table)
            df.at[column1, column2] = f"{result.statistic}, {result.pvalue}"
            df.at[column2, column1] = f"{result.statistic}, {result.pvalue}"

        if result.pvalue < ALPHA:
                # print(f"Results for {column1} and {column2} are significantly different.")
                # print("McNemar Test Statistic:", result.statistic)
                # print("McNemar Test P-value:", result.pvalue)
                test_results.append((column1, column2, result.statistic, result.pvalue))
        df.to_csv(f"{self.results_folder}/mcnemar_results.csv")

        # Print the results
        # for result in test_results:
        #     print(f"Results for {result[0]} and {result[1]} are significantly different.")
        #     print("McNemar Test Statistic:", result[2])
        #     print("McNemar Test P-value:", result[3])

    def load_models(self) -> list:
        """
        Load the models from the specified path.
        @return: The models
        """
        models = json.load(open(f"{self.results_folder}/models.json"))
        return models

    def load_datasets(self) -> list:
        """
        Load the datasets from the specified path.
        @return: The datasets
        """
        datasets = json.load(open(f"{self.results_folder}/datasets.json"))
        return datasets


if __name__ == "__main__":
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, choices=LLMProcessorConstants.MODEL_NAMES.keys(),
                      default=list(LLMProcessorConstants.MODEL_NAMES.keys())[1])
    args.add_argument("--dataset", type=str, choices=DatasetsConstants.DATASET_NAMES,
                      default=DatasetsConstants.DATASET_NAMES[0])
    args = args.parse_args()
    model = LLMProcessorConstants.MODEL_NAMES[args.model].split('/')[-1]
    # Perform McNemar test
    mcnemar_test = McNemarTest(model, args.dataset)
    results = mcnemar_test.load_results()
    mcnemar_test.perform_mcnemar_test(results)
    # mcnemar.perform_cochrans_q_test(results)

