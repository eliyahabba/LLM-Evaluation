import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.Analysis.ModelDatasetRunner import ModelDatasetRunner
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.Visualization.ChooseBestCombination import ChooseBestCombination
from src.Analysis.PerformAnalysis import PerformAnalysis
from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
DatasetsConstants = Constants.DatasetsConstants
ResultConstants = Constants.ResultConstants

file_path = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH / f"{ResultConstants.BEST_COMBINATIONS}.csv"


class BestCasualParamsFinder:
    def __init__(self,
                 dataset_file_name: str,
                 performance_summary_path: Path,
                 comparison_matrix_path: Path,
                 selected_best_value_axes: list):
        self.dataset_file_name = dataset_file_name
        self.performance_summary_path = performance_summary_path
        self.comparison_matrix_path = comparison_matrix_path
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

    def find_stats_test(self, grouped_metadata_df: pd.DataFrame, best_row: pd.Series):
        perform_analysis = PerformAnalysis(self.comparison_matrix_path, grouped_metadata_df, best_row)
        return perform_analysis.calculate_cochrans_q_test()

    @staticmethod
    def find_best_casual_params(format_folder: Path, eval_value: str):
        """
        Find the best row in the performance_summary_df.
        """
        performance_summary_path = format_folder / f"{ResultConstants.PERFORMANCE_SUMMARY}_{eval_value}_data.csv"
        comparison_matrix_path = format_folder / f"{ResultConstants.COMPARISON_MATRIX}_{eval_value}_data.csv"
        dataset_file_name = format_folder.parents[1].name
        selected_best_value_axes = list(ConfigParams.override_options.keys())
        perform_analysis_runner = BestCasualParamsFinder(dataset_file_name, performance_summary_path,
                                                         comparison_matrix_path,
                                                         selected_best_value_axes)
        grouped_metadata_df, best_row = perform_analysis_runner.get_best_row_and_grouped_metadata_df()
        result = perform_analysis_runner.find_stats_test(grouped_metadata_df, best_row)
        # add to the best_row the result.stats and result.pvalue
        stats_values = pd.Series({'statistic': f'{result.statistic:.2f}', 'pvalue': f'{result.pvalue:.2f}'})
        best_row = pd.concat([best_row, stats_values])
        if file_path.exists():
            best_combinations_df = pd.read_csv(file_path, dtype=str)
        else:
            best_combinations_df = pd.DataFrame()

        # create a new df with the best row, and add 2 columns: model, dataset
        columns = ['model', 'dataset'] + best_row.index.tolist()
        # Create a list of values for each row
        data = [[format_folder.parents[2].name, dataset_file_name] + best_row.tolist()]
        # Create the DataFrame
        if best_combinations_df.empty:
            best_combinations_df = pd.DataFrame(data, columns=columns)
        else:
            condition = (best_combinations_df['model'] == format_folder.parents[2].name) & (
                    best_combinations_df['dataset'] == dataset_file_name)
            if condition.any():
                id = best_combinations_df[condition].index[0]
                for key, value in best_row.items():
                    best_combinations_df.at[id, key] = value
            else:
                # add the new row to the DataFrame (not use append)
                best_combinations_df = pd.concat([best_combinations_df, pd.DataFrame(data, columns=columns)], ignore_index=True)
        best_combinations_df.to_csv(file_path, index=False)
        return best_row


if __name__ == "__main__":
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=DatasetsConstants.DATASET_NAMES,
                      default=DatasetsConstants.DATASET_NAMES[0])

    args = args.parse_args()
    # Load the model and the dataset
    results_folder = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH
    eval_on = ExperimentConstants.EVALUATE_ON
    model_dataset_runner = ModelDatasetRunner(results_folder, eval_on)
    model_dataset_runner.run_function_on_all_models_and_datasets(BestCasualParamsFinder.find_best_casual_params)
