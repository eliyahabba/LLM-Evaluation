import json
from pathlib import Path

import pandas as pd

from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants



class ExperimentsResultsFolder:
    def __init__(self, eval_on: str):
        self.eval_on = eval_on

    def load_experiment_file(self, results_file: Path):
        """
        Load the results from the json file.
        @return: list of results
        """
        with open(results_file, "r") as f:
            json_data = json.load(f)
        self.experiment = json_data

    def count_results(self, results_file: Path):
        """
        Count the results of the model on the dataset.
        @return:
        """
        self.load_experiment_file(results_file)
        result_counter = {}
        results = self.experiment['results']
        for eval_on_value in self.eval_on:
            if eval_on_value not in results:
                result_counter[eval_on_value] = 0
                continue
            results_to_eval = results[eval_on_value]
            predictions_idx = [result['Index'] for result in results_to_eval]
            result_counter[eval_on_value] = len(predictions_idx)

        return result_counter

def check_results_files(format_folder):
    results_files = list([file for file in format_folder.glob("*.json")])
    sorted_file_paths = sorted(results_files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

    experiments_results = ExperimentsResultsFolder(eval_on)
    for results_file in sorted_file_paths:
        try:
            results_counter = experiments_results.count_results(results_file)
            # if one of the values of results_counter < 100 then print the results_counter
            if any([value < 100 for value in results_counter.values()]):
                print(f"{results_file}: {results_counter}")
        except Exception as e:
            print(f"Error in {results_file}: {e}")
            continue

def check_comparison_matrix(format_folder: Path, eval_value: str):
    df = pd.read_csv(format_folder/ f"comparison_matrix_{eval_value}_data.csv")
    # read the columns of the dataframe
    columns = df.columns
    # for each column in the dataframe, if the value in the columns is less than 100 or contains NaN, print the column
    for column in columns:
        if df[column].isnull().values.any() or len(df[column]) < 100:
            print(f"{format_folder} in {column}: Missing {df[column].isnull().sum()} values")

if __name__ == "__main__":
    # Load the model and the dataset
    results_folder = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH
    eval_on = ExperimentConstants.EVALUATE_ON
    models_names = sorted([file for file in results_folder.glob("*") if file.is_dir()], key=lambda x: x.name.lower())
    for model_name in models_names:
        datasets = sorted([file for file in model_name.glob("*") if file.is_dir()])
        for dataset_folder in datasets:
            shots = [file for file in dataset_folder.glob("*") if file.is_dir()]
            for shot in shots:
                formats = [file for file in shot.glob("*") if file.is_dir()]
                for format_folder in formats:
                    for eval_value in eval_on:
                        check_comparison_matrix(format_folder, eval_value)
                    # check_results_files(format_folder)
