import sys
from pathlib import Path

import pandas as pd
import streamlit as st

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))

from src.Visualization.ResultsLoader import ResultsLoader
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams

class VisualizeResults:
    def display_page(self):
        st.title("Templates Visualization")
        dataset_file_name, selected_shot_file_name = ResultsLoader.select_dataset_and_shot()
        self.display_possible_templates_args()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, result_file = ResultsLoader.select_result_file(result_files, "scores")

        self.display_results(result_file)

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
            for k, v in override_options.items():
                # display the options in the main page
                st.markdown(f"The possible values for **{k}** are:  \n")
                st.markdown(f"**{v}**")


if __name__ == "__main__":
    vr = VisualizeResults()
    vr.display_page()
