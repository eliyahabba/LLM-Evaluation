import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from src.Analysis.ModelDatasetRunner import ModelDatasetRunner
from src.Analysis.StatisticalTests.CompareSeriesBinaryDataFromTable import CompareSeriesBinaryDataFromTable
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.Visualization.ChooseBestCombination import ChooseBestCombination
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
DatasetsConstants = Constants.DatasetsConstants
ResultConstants = Constants.ResultConstants


class PerformAnalysisRunner:
    def __init__(self,
                 dataset_file_name: str,
                 performance_summary_path: Path,
                 selected_best_value_axes: list):
        self.dataset_file_name = dataset_file_name
        self.performance_summary_path = performance_summary_path
        self.selected_best_value_axes = selected_best_value_axes

    def get_best_row_and_grouped_metadata_df(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the best row and the grouped metadata dataframe.
        @return:
        """
        choose_best_combination = ChooseBestCombination(self.dataset_file_name, self.performance_summary_path,
                                                        self.selected_best_value_axes)
        grouped_metadata_df, best_row = choose_best_combination.choose_best_combination()
        return grouped_metadata_df, best_row


def find_best_casual_params(format_folder: Path, eval_value: str):
    """
    Find the best row in the performance_summary_df.
    """
    performance_summary_path = format_folder / f"{ResultConstants.PERFORMANCE_SUMMARY}_{eval_value}_data.csv"
    dataset_file_name = format_folder.parents[1].name
    selected_best_value_axes = list(ConfigParams.override_options.keys())
    perform_analysis_runner = PerformAnalysisRunner(dataset_file_name, performance_summary_path,
                                                    selected_best_value_axes)
    grouped_metadata_df, best_row = perform_analysis_runner.get_best_row_and_grouped_metadata_df()
    # create a new df with the best row, and add 2 columns: model, dataset
    columns = ['model', 'dataset'] + best_row.index.tolist()
    # Create a list of values for each row
    data = [[format_folder.parents[2].name, dataset_file_name] + best_row.tolist()]
    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)
    st.write(df)



if __name__ == "__main__":
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, choices=LLMProcessorConstants.MODEL_NAMES.keys(),
                      default=list(LLMProcessorConstants.MODEL_NAMES.keys())[1])
    args.add_argument("--dataset", type=str, choices=DatasetsConstants.DATASET_NAMES,
                      default=DatasetsConstants.DATASET_NAMES[0])
    args = args.parse_args()
    model = LLMProcessorConstants.MODEL_NAMES[args.model].split('/')[-1]
    # Load the model and the dataset
    results_folder = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH
    eval_on = ExperimentConstants.EVALUATE_ON
    model_dataset_runner = ModelDatasetRunner(results_folder, eval_on)
    model_dataset_runner.run_function_on_all_models_and_datasets(find_best_casual_params)
