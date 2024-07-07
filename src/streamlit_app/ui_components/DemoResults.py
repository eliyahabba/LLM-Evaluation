import sys
from pathlib import Path

import pandas as pd
import streamlit as st

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))
from src.streamlit_app.ui_components.SelectAxes import SelectAxes
from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
from src.streamlit_app.ui_components.ChooseBestCombination import ChooseBestCombination
from src.streamlit_app.ui_components.CreateHeatmap import CreateHeatmap
from src.streamlit_app.ui_components.AnalysisDisplay import AnalysisDisplay
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants


class VisualizeResults:
    def display_page(self):
        st.title("Templates Visualization")
        dataset_file_name, selected_shot_file_name, _ = ResultsLoader.select_experiment_params()
        self.display_possible_templates_args()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, performance_summary_path = ResultsLoader.select_result_file(result_files,
                                                                                      ResultConstants.
                                                                                      PERFORMANCE_SUMMARY)
        comparison_matrix_path = \
            [result_file for result_file in result_files if ResultConstants.COMPARISON_MATRIX in result_file.name][0]
        # with st.expander("The results of the model"):
        #     self.display_results(performance_summary_path)
        # self.select_and_display_best_combination(dataset_file_name, performance_summary_path, comparison_matrix_path)
        # self.display_heatmap(dataset_file_name, performance_summary_path)
        ResultsLoader.display_sample_examples(selected_shot_file_name, dataset_file_name, result_file_name)

    def display_results(self, results_file: Path):
        """
        Display the results of the model.

        @param results_file: the path to the results file
        @return: None
        """
        df = pd.read_csv(results_file)
        cols_to_remove = ['card', 'system_format', 'score', 'score_name']
        if 'groups_mean_score' in df.columns:
            cols_to_remove.append('groups_mean_score')
        columns_order = ['template_name', 'number_of_instances', 'accuracy', 'accuracy_ci_low', 'accuracy_ci_high',
                         'score_ci_low', 'score_ci_high']
        df = df.drop(cols_to_remove, axis=1)
        df = df[columns_order]
        st.write(df)

    def display_possible_templates_args(self):
        with st.expander("Possible template arguments"):
            override_options = ConfigParams.override_options
            color_palette = ["#FF5733", "#3366FF", "#FF33E9", "#4CAF50"]  # Soft green color added

            for i, (k, v) in enumerate(override_options.items()):
                # Choose a color from the color palette based on the index
                color = color_palette[i % len(color_palette)]

                # Display the options in the main page with different colors
                st.markdown(f'<span style="color:{color}">The possible values for <b>{k}</b> are:</span>',
                            unsafe_allow_html=True)
                st.markdown(f'<span style="color:{color}"><b>{v}</b></span>', unsafe_allow_html=True)

    def select_and_display_best_combination(self, dataset_file_name: str, performance_summary_path: Path,
                                            comparison_matrix_path: Path):
        select_axes = SelectAxes()
        selected_best_value_axes = select_axes.select_causal_axes()
        choose_best_combination = ChooseBestCombination(dataset_file_name, performance_summary_path,
                                                        selected_best_value_axes)
        grouped_metadata_df, best_row = choose_best_combination.choose_best_combination()
        select_axes.write_best_combination(best_row)

        perform_analysis = AnalysisDisplay(comparison_matrix_path, grouped_metadata_df, best_row)
        # ask the uset tpo select the top K results that hw want to compare
        perform_analysis.display_cochrans_q_test()
        perform_analysis.display_mcnemar_test(best_row)

    def display_heatmap(self, dataset_file_name: str, performance_summary_path: Path):
        # add an expander to the heatmap
        with st.expander("Heatmap of the accuracy of the templates"):
            create_heatmap = CreateHeatmap(dataset_file_name, performance_summary_path)
            create_heatmap.create_heatmap()


if __name__ == "__main__":
    vr = VisualizeResults()
    vr.display_page()
