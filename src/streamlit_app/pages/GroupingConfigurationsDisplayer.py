import sys
from pathlib import Path

import streamlit as st

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))

from src.streamlit_app.ui_components.GroupingConfigurations import GroupingConfigurations
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants


class GroupingConfigurationsDisplayer:
    def __init__(self):
        self.grouping_configurations = GroupingConfigurations()

    def display_page(self):
        st.title("Grouping The Configurations")
        dataset_file_name, selected_shot_file_name, _ = ResultsLoader.select_experiment_params()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, performance_summary_path = ResultsLoader.select_result_file(result_files,
                                                                                      ResultConstants.
                                                                                      PERFORMANCE_SUMMARY)
        comparison_matrix_path = \
            [result_file for result_file in result_files if ResultConstants.COMPARISON_MATRIX in result_file.name][0]
        results = self.grouping_configurations.select_and_display_best_combination(dataset_file_name,
                                                                                   performance_summary_path,
                                                                                   comparison_matrix_path)
        st.write(results)


if __name__ == "__main__":
    grouping_configurations_displayer = GroupingConfigurationsDisplayer()
    grouping_configurations_displayer.display_page()
