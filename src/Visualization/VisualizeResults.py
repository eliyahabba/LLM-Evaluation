import sys
from pathlib import Path

import pandas as pd
import streamlit as st

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))

from src.Visualization.ResultsLoader import ResultsLoader
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.Visualization.CreateHeatmap import CreateHeatmap


class VisualizeResults:
    def display_page(self):
        st.title("Templates Visualization")
        dataset_file_name, selected_shot_file_name = ResultsLoader.select_experiment_params()
        self.display_possible_templates_args()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, result_file = ResultsLoader.select_result_file(result_files, "scores")

        self.display_results(result_file)
        self.display_heatmap(dataset_file_name, result_file)
        ResultsLoader.display_sample_examples(selected_shot_file_name, dataset_file_name, result_file_name)

    def display_results(self, results_file: Path):
        """
        Display the results of the model.

        @param results_file: the path to the results file
        @return: None
        """
        df = pd.read_csv(results_file)
        cols_to_remove = ['card', 'system_format', 'score', 'score_name']
        columns_order = ['template_name', 'number_of_instances', 'accuracy', 'accuracy_ci_low', 'accuracy_ci_high',
                         'score_ci_low', 'score_ci_high', 'groups_mean_score']
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

    def display_heatmap(self, dataset_file_name: str, result_file: Path):
        create_heatmap = CreateHeatmap(dataset_file_name, result_file)
        create_heatmap.create_heatmap()


if __name__ == "__main__":
    vr = VisualizeResults()
    vr.display_page()
