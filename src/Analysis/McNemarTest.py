import argparse

import pandas as pd
from tqdm import tqdm

from src.Analysis.McNemarTestFromTable import McNemarTestFromTable
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
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


class McNemarTest:
    """
    Class to perform McNemar test on the results that are obtained from different models and datasets.
    """

    def __init__(self, model: str, dataset: str,
                 main_results_folder: str = MAIN_RESULTS_PATH):
        self.main_results_folder = main_results_folder
        self.results_folder = f"{main_results_folder}/{RESULTS_FOLDER}/{model}/{dataset}/{SHOT}/{FORMAT}"

    def load_results(self, eval_value: str = TRAIN_OR_TEST_TYPE) -> pd.DataFrame:
        """
        Load the results from the specified path.
        @return: The results
        """
        file_name = f"{ResultConstants.COMPARISON_MATRIX}_{eval_value}_data.csv"
        file_path = f"{self.results_folder}/{file_name}"
        results = pd.read_csv(file_path)
        return results

    @staticmethod
    def perform_cochrans_q_test(results):
        """
        Perform Cochran's Q test on the given results.
        """
        # Perform Cochran's Q test
        result = McNemarTestFromTable.perform_cochrans_q_test_from_table(results)
        # Print results
        print("Cochran's Q Test Statistic:", result.statistic)
        print(f"Cochran's Q Test P-value:{result.pvalue}")

    @staticmethod
    def perform_mcnemar_test(results: pd.DataFrame):
        """
        Perform McNemar test on the given results.
        @param results1: Results from model 1
        @param results2: Results from model 2
        """
        df = McNemarTestFromTable.perform_mcnemar_test_from_table(results)

        # if result.pvalue < McNemarTestConstants.ALPHA:
        #     # print(f"Results for {column1} and {column2} are significantly different.")
        #     # print("McNemar Test Statistic:", result.statistic)
        #     # print("McNemar Test P-value:", result.pvalue)
        #     test_results.append((column1, column2, result.statistic, result.pvalue))
        return df

    def save_mcnemar_resuls(self, df):
        df.to_csv(f"{self.results_folder}/mcnemar_results.csv")

        # Print the results
        # for result in test_results:
        #     print(f"Results for {result[0]} and {result[1]} are significantly different.")
        #     print("McNemar Test Statistic:", result[2])
        #     print("McNemar Test P-value:", result[3])


def run_all():
    for model_key, model_name in tqdm(LLMProcessorConstants.MODEL_NAMES.items()):
        model = Utils.get_model_name(model_name)
        for dataset in tqdm(DatasetsConstants.DATASET_NAMES):
            mcnemar_test = McNemarTest(model, dataset)
            results = mcnemar_test.load_results()
            mcnemar_test.perform_mcnemar_test(results)
            mcnemar_test.perform_cochrans_q_test(results)


if __name__ == "__main__":
    # run_all()
    # exit()
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
    # df = mcnemar_test.perform_mcnemar_test(results)
    # mcnemar_test.save_mcnemar_resuls(df)
    mcnemar_test.perform_cochrans_q_test(results)
