import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.Analysis.ModelDatasetRunner import ModelDatasetRunner
from src.utils.Constants import Constants
from src.utils.MMLUData import MMLUData

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
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
        try:
            self.load_experiment_file(results_file)
        except Exception as e:
            print(f"Error loading file {results_file}: {e}")
            return 0
        results = self.experiment['results']
        if self.eval_on not in results:
            return 0
        result_counter = len(results[self.eval_on])
        return result_counter

    def get_data_percentage_and_completed(self, results_folder, model, mmlu_dataset, shots):
        # def print_future_experiments(format_folder: Path, eval_value: str, kwargs: dict = None):
        path = Path(results_folder) / model / mmlu_dataset / shots / "empty_system_format"
        results_files = list([file for file in path.glob("*.json")])
        sorted_file_paths = sorted(results_files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

        experiments_results = ExperimentsResultsFolder(eval_on)
        # model_name = sorted_file_paths[0].parents[3].name.split("-")[0].lower()
        mmlu_dataset_sizes = pd.read_csv(TemplatesGeneratorConstants.MMLU_DATASET_SIZES_PATH)
        dataset_size = mmlu_dataset_sizes[mmlu_dataset_sizes["Name"] == mmlu_dataset.split("mmlu.")[1]]

        all_files_names = [f"experiment_template_{i}" for i in range(0, 56)]
        sorted_file_names = [file.name.split(".json")[0] for file in sorted_file_paths]

        missing_files = [file for file in all_files_names if file not in sorted_file_names]
        files_total_data = {}
        files_acquired_data = {}
        for missing_file in missing_files:
            configuration_number = int(missing_file.split("experiment_template_")[1])
            total_data = dataset_size.iloc[0].to_dict()[self.eval_on]
            data_acquired = 0
            files_total_data[configuration_number] = total_data
            files_acquired_data[configuration_number] = data_acquired

        for results_file in sorted_file_paths:
            results_counter = experiments_results.count_results(results_file)
            # if one of the values of results_counter < 100 then print the results_counter
            configuration_number = int(results_file.name.split("experiment_template_")[1].split(".")[0])
            # percentage_of_evaluated_data = (results_counter // dataset_size.iloc[0].to_dict()[self.eval_on]) * 100
            files_total_data[configuration_number] = dataset_size.iloc[0].to_dict()[self.eval_on]
            files_acquired_data[configuration_number] = results_counter
        return files_total_data, files_acquired_data


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


def create_summarize_df():
    names = ['Model', 'Dataset', 'Shots', "Configuration"]
    index = pd.MultiIndex(levels=[[], [], [], []], codes=[[], [], [], []], names=names)
    df = pd.DataFrame(index=index, columns=['Total Data', 'Data Acquired'])
    # add a dummmy row to the df with the update_function
    df.loc[('dummy', 'dummy', 'dummy', "dummy"), ['Total Data', 'Data Acquired']] = [0, False]
    return df


def create_save_summarize_df(file_path):
    df = create_summarize_df()
    # df.index = df.index.set_levels(df.index.levels[3].astype(int), level=3)
    df.to_csv(file_path)


def get_summarize_df(file_path):
    if not file_path.exists():
        create_save_summarize_df(file_path)
    return pd.read_csv(file_path, index_col=[0, 1, 2, 3])


def udpate_summarize_df(df, model, dataset, shots, config_number, total_data, data_acquired):
    df.sort_index(inplace=True)
    if (model, dataset, shots, config_number) not in df.index:
        # add the new row
        new_row = pd.DataFrame({
            'Total Data': [total_data],
            'Data Acquired': [data_acquired],
        }, index=pd.MultiIndex.from_tuples([(model, dataset, shots, config_number)],
                                           names=['Model', 'Dataset', 'Shots', "Configuration"]))
        df = pd.concat([df, new_row])
    else:
        df.loc[(model, dataset, shots, config_number), ['Total Data', 'Data Acquired']] = [
            total_data, data_acquired]

    return df


def save_updated_summarize_df(df, file_path):
    df.to_csv(file_path)


def update_df_files(files_total_data, files_acquired_data, df, model_name, mmlu_dataset, shots):
    for config_number, total_data in files_total_data.items():
        data_acquired = files_acquired_data[config_number]
        df = udpate_summarize_df(df, model_name, mmlu_dataset, shots, config_number,
                                 total_data, data_acquired)
    return df


if __name__ == "__main__":
    # Load the model and the dataset
    results_folder = ExperimentConstants.MAIN_RESULTS_PATH / Path(
        TemplatesGeneratorConstants.MULTIPLE_CHOICE_INSTRUCTIONS_FOLDER_NAME)
    results_folder = ExperimentConstants.MAIN_RESULTS_PATH / Path(
        TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME)
    eval_on = ExperimentConstants.EVALUATE_ON_ANALYZE[0]
    experiments_results = ExperimentsResultsFolder(eval_on)
    model_dataset_runner = ModelDatasetRunner(results_folder, eval_on)
    # create a df
    df = get_summarize_df(ResultConstants.SUMMARIZE_DF_PATH)
    MMLUData.initialize()
    # run the function on all the models and datasets
    for model in tqdm(list(LLMProcessorConstants.MODEL_NAMES.values())):
        model_name = model.split("/")[-1]
        for mmlu_dataset in tqdm(list(MMLUData.get_mmlu_datasets())):
            for shots in ResultConstants.SHOTS:
                files_total_data, files_acquired_data = experiments_results. \
                    get_data_percentage_and_completed(results_folder, model_name, mmlu_dataset, shots)
                df = update_df_files(files_total_data, files_acquired_data, df, model_name, mmlu_dataset, shots)
                save_updated_summarize_df(df, ResultConstants.SUMMARIZE_DF_PATH)
