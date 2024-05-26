import argparse
import sys
from pathlib import Path

import pandas as pd

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))

from src.Analysis.ModelDatasetRunner import ModelDatasetRunner
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.Visualization.ChooseBestCombination import ChooseBestCombination
from src.Visualization.AnalysisDisplay import AnalysisDisplay
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
DatasetsConstants = Constants.DatasetsConstants
ResultConstants = Constants.ResultConstants


class GroupingConfigurations:
    @staticmethod
    def assign_groups(df, best_row, threshold=0.05):
        # Initialize a new column for grouping
        best_row_mask = df.index == f'best set: {best_row.template_name[0]}'
        # take from the df only the column that is the best row
        best_row_in_series = df[best_row_mask].squeeze()

        # remove the best row from the df
        def filter_by_pvalue(x):
            if x is None:  # Handle the None case separately
                return False
            return True if x == "None" else float(x.split(',')[1]) > threshold

        # Apply the filter to the Series to create a filtered Series
        similar_rows = best_row_in_series[best_row_in_series.apply(filter_by_pvalue)]
        # take all the index of values that the [1] value is above the threshold
        # Identify all rows similar to the best row based on the given threshold (except the best row itself)
        # find all the columns in best_row_in_df (it is one row) are within the threshold

        # Move to the next group label
        return similar_rows.index

    @staticmethod
    def select_and_display_best_combination(dataset_file_name: str, performance_summary_path: Path,
                                            comparison_matrix_path: Path):
        current_group = 'A'
        exclude_templates = []
        origin_results = None
        for i in range(5):
            choose_best_combination = ChooseBestCombination(dataset_file_name, performance_summary_path,
                                                            selected_best_value_axes=list(
                                                                ConfigParams.override_options.keys()))
            grouped_metadata_df, best_row = choose_best_combination.choose_best_combination(
                exclude_templates=exclude_templates)
            perform_analysis = AnalysisDisplay(comparison_matrix_path, grouped_metadata_df, best_row,
                                               ask_on_key=False,
                                               default_top_k=56)

            results = perform_analysis.get_results_of_mcnemar_test(best_row=best_row)
            if origin_results is None:
                origin_results = results
            if len(results) == 1 and results.index[0] == f'best set: {best_row.template_name[0]}':
                origin_results.loc[best_row.template_name[0], 'group'] = current_group
                break
            else:
                similar_rows = GroupingConfigurations.assign_groups(results, best_row)
                results.loc[similar_rows, 'group'] = current_group
                # take only the similar_rows from the results
                results = results.loc[similar_rows]
                exclude_templates.extend(similar_rows)
            current_group = chr(ord(current_group) + 1)
            # remove the best row from the exclude_templates
            best_row_template = f'best set: {best_row.template_name[0]}'
            if best_row_template in exclude_templates:
                del exclude_templates[exclude_templates.index(best_row_template)]
            exclude_templates.append(best_row.template_name[0])
            def process_index(index):
                """Process each index value to retain only the relevant part."""
                return [x.split(' ')[2] if 'best set: ' in x else x for x in index]

            if i != -1:
                # remove the "best set: " from the template name in the df
                results.index = pd.Index(process_index(results.index))
                # remove the
            if 'group' not in origin_results.columns:
                origin_results['group'] = None
            index_name = origin_results.index.name
            origin_results.index = pd.Index(process_index(origin_results.index))
            origin_results.index.name = index_name

            origin_results.loc[results.index, 'group'] = results['group']
            # take the template column, every row is list, and flatten it
            grouped_metadata_df['template_name'] = grouped_metadata_df['template_name'].apply(lambda x: x[0])
            grouped_metadata_df.set_index('template_name', inplace=True)
            grouped_metadata_df.index = pd.Index(process_index(grouped_metadata_df.index))

            origin_results.loc[grouped_metadata_df.index, 'accuracy'] = grouped_metadata_df['accuracy']
            if len(exclude_templates) >= 56:
                break
        return origin_results

    @staticmethod
    def get_group_for_dataset(format_folder: Path, eval_value: str,
                              kwargs: dict = None
                              ) -> None:
        """
        Find the best row in the performance_summary_df.
        """
        performance_summary_path = format_folder / f"{ResultConstants.PERFORMANCE_SUMMARY}_{eval_value}_data.csv"
        comparison_matrix_path = format_folder / f"{ResultConstants.COMPARISON_MATRIX}_{eval_value}_data.csv"
        dataset_file_name = format_folder.parents[1].name

        grouped_result_file = format_folder / f"{ResultConstants.GROUPED_LEADERBOARD}.csv"
        if not grouped_result_file.exists() or True:
            grouped_metadata_df = GroupingConfigurations.select_and_display_best_combination(dataset_file_name,
                                                                                             performance_summary_path,
                                                                                             comparison_matrix_path)
            # Save the grouped_metadata_df to a file
            grouped_metadata_df.to_csv(grouped_result_file, index=True)
            print(f"Saved the grouped metadata to {grouped_result_file}")


if __name__ == "__main__":
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=DatasetsConstants.DATASET_NAMES,
                      default=DatasetsConstants.DATASET_NAMES[0])
    args.add_argument("--results_folder", type=str, default=TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME)
    args = args.parse_args()
    args.results_folder = ExperimentConstants.MAIN_RESULTS_PATH / Path(args.results_folder)
    # Load the model and the dataset
    eval_on = ExperimentConstants.EVALUATE_ON_ANALYZE
    model_dataset_runner = ModelDatasetRunner(args.results_folder, eval_on)
    grouping_configurations = GroupingConfigurations()

    model_dataset_runner.run_function_on_all_models_and_datasets(grouping_configurations.get_group_for_dataset)
