import sys
from pathlib import Path

import pandas as pd
import streamlit as st

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.Visualization.ChooseBestCombination import ChooseBestCombination
from src.Visualization.AnalysisDisplay import AnalysisDisplay
from src.Visualization.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants


class GroupingConfigurations:
    def display_page(self):
        st.title("Grouping The Configurations")
        dataset_file_name, selected_shot_file_name = ResultsLoader.select_experiment_params()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, performance_summary_path = ResultsLoader.select_result_file(result_files,
                                                                                      ResultConstants.
                                                                                      PERFORMANCE_SUMMARY)
        comparison_matrix_path = \
            [result_file for result_file in result_files if ResultConstants.COMPARISON_MATRIX in result_file.name][0]
        self.select_and_display_best_combination(dataset_file_name, performance_summary_path, comparison_matrix_path)

    def assign_groups(self, df, best_row, current_group, threshold=0.05):
        # Initialize a new column for grouping
        # df['group'] = None
        best_row_mask = df.index == f'best set: {best_row.template_name[0]}'
        # take from the df only the column that is the best row
        best_row_in_series = df[best_row_mask].squeeze()
        # best_row_in_series = best_row_in_series[~best_row_mask]

        # remove the best row from the df
        def filter_by_pvalue(x):
            if x is None:  # Handle the None case separately
                return False
            return True if x=="None" else float(x.split(',')[1]) > threshold

        # Apply the filter to the Series to create a filtered Series
        similar_rows = best_row_in_series[best_row_in_series.apply(filter_by_pvalue)]
        # take all the index of values that the [1] value is above the threshold
        # Identify all rows similar to the best row based on the given threshold (except the best row itself)
        # find all the columns in best_row_in_df (it is one row) are within the threshold

        # Move to the next group label
        return similar_rows.index

    def select_and_display_best_combination(self, dataset_file_name: str, performance_summary_path: Path,
                                            comparison_matrix_path: Path):
        current_group = 'A'
        exclude_templates = []
        curr_results = pd.DataFrame()
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
            # ask the uset tpo select the top K results that hw want to compare
            results = perform_analysis.get_results_of_mcnemar_test(best_row=best_row)
            if origin_results is None:
                origin_results = results
            similar_rows = self.assign_groups(results, best_row, current_group)
            results.loc[similar_rows, 'group'] = current_group
            # take only the similar_rows from the results
            results = results.loc[similar_rows]
            current_group = chr(ord(current_group) + 1)
            exclude_templates.extend(similar_rows)
            exclude_templates.append(best_row.template_name[0])
            if i != 0:
                # remove the "best set: " from the template name in the df
                def process_index(index):
                    """Process each index value to retain only the relevant part."""
                    return [x.split(' ')[2] if 'best set: ' in x else x for x in index]

                results.index = pd.Index(process_index(results.index))
                # remove the
            if 'group' not in origin_results.columns:
                origin_results['group'] = None
            origin_results.loc[results.index, 'group'] = results['group']
            if len(exclude_templates) >= 56:
                break

        st.write(origin_results)


if __name__ == "__main__":
    grouping_configurations = GroupingConfigurations()
    grouping_configurations.display_page()
