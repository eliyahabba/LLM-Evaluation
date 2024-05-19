import json
from pathlib import Path

import pandas as pd

from src.Analysis.ModelDatasetRunner import ModelDatasetRunner
from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants

mmlu_dataset_sizes = pd.read_csv(TemplatesGeneratorConstants.MMLU_DATASET_SIZES_PATH)


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


def print_future_experiments(format_folder: Path, eval_value: str, kwargs: dict = None):
    results_files = list([file for file in format_folder.glob("*.json")])
    sorted_file_paths = sorted(results_files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

    experiments_results = ExperimentsResultsFolder(eval_on)
    model_name = sorted_file_paths[0].parents[3].name.split("-")[0].lower()
    if "llama" in model_name:
        # check if it 7B o3 13B or 70B and add the correct model name
        params_num = sorted_file_paths[0].parents[3].name.split("-")[2].lower()
        assert "b" in params_num
        model_name = f"{model_name}{params_num.split('b')[0]}"
    dataset_name = sorted_file_paths[0].parents[2].name
    if "mmlu" not in dataset_name:
        return
    mmlu_dataset_sizes = pd.read_csv(TemplatesGeneratorConstants.MMLU_DATASET_SIZES_PATH)
    dataset_size = mmlu_dataset_sizes[mmlu_dataset_sizes["Category"] == dataset_name.split("mmlu.")[1]]
    exs_numbers = []
    all_files_names = [f"experiment_template_{i}" for i in range(0, 56)]
    sorted_file_names = [file.name.split(".json")[0] for file in sorted_file_paths]
    missing_files = [file for file in all_files_names if file not in sorted_file_names]
    for missing_file in missing_files:
        exs = int(missing_file.split("experiment_template_")[1])
        exs_numbers.append(exs)
    for results_file in sorted_file_paths:
        try:
            results_counter = experiments_results.count_results(results_file)
            # if one of the values of results_counter < 100 then print the results_counter
            exs = int(results_file.name.split("experiment_template_")[1].split(".")[0])
            if any([results_counter[key] < dataset_size.iloc[0].to_dict()[key] for key
                    in results_counter.keys()]):
                exs_numbers.append(exs)
                # print(f"sbatch {model_name} /run_mmlu.sh cards.{dataset_name} 0 56")
        except Exception as e:
            print(f"Error in {results_file}: {e}")
            continue
    if exs_numbers:
        min_exs = min(exs_numbers)
        max_exs = max(exs_numbers) + 1
        # range(min_exs, max_exs, 10)
        for i in range(min_exs, max_exs, 10):
            end = i + 10 if i + 10 < max(exs_numbers) else max(exs_numbers)
            # print(f"sbatch {model_name}/run_mmlu.sh cards.{dataset_name} {i} {min(56,end)}"
            #      f" ;")
        # for i in exs_numbers:
        #     print(f"sbatch {model_name}/run_mmlu.sh cards.{dataset_name} {i} {i + 1}"
        #           f" ;")

    print(f"sbatch {model_name}/run_mmlu.sh cards.{dataset_name} {0} {28};")
    print(f"sbatch {model_name}/run_mmlu.sh cards.{dataset_name} {28} {56};")

def check_comparison_matrix(format_folder: Path, eval_value: str, kwargs: dict = None):
    if not format_folder.exists():
        print(f"{format_folder} does not exist")
    try:
        df = pd.read_csv(format_folder / f"{ResultConstants.COMPARISON_MATRIX}_{eval_value}_data.csv")
    except Exception as e:
        print(f"Error in {format_folder}: {e}")
        return

    # read the columns of the dataframe
    columns = df.columns
    # for each column in the dataframe, if the value in the columns is less than 100 or contains NaN, print the column
    for column in columns:
        if df[column].isnull().values.any() or len(df[column]) < 100:
            print(f"{format_folder} in {column}: Missing {df[column].isnull().sum()} values")
    all_gold_columns_names = [f"experiment_template_{i}" for i in range(0, 56)]
    for gold_column in all_gold_columns_names:
        if gold_column not in columns:
            print(f"{format_folder} does not contain {gold_column}")


if __name__ == "__main__":
    # Load the model and the dataset
    results_folder = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH
    eval_on = ExperimentConstants.EVALUATE_ON_ANALYZE
    model_dataset_runner = ModelDatasetRunner(results_folder, eval_on)
    # model_dataset_runner.run_function_on_all_models_and_datasets(check_comparison_matrix)
    model_dataset_runner.run_function_on_all_models_and_datasets(print_future_experiments)
