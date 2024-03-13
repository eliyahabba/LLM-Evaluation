import json
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import streamlit as st

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants

results_folder = ExperimentConstants.RESULTS_PATH


class VisualizeResults:
    def display_page(self):
        st.title("Model Visualization")

        results_files = [f for f in results_folder.iterdir() if f.is_file() and f.name.startswith("cards")]
        results_names_to_display = {f.name.split("_scores.")[-2]: f for f in results_files}

        result_file_name = st.sidebar.selectbox("Select dataset to visualize", list(results_names_to_display.keys()))
        results_file = results_names_to_display[result_file_name]
        self.display_results(results_file)
        self.display_sample_examples(result_file_name)

    def display_results(self, results_file: Path):
        """
        Display the results of the model.

        @param results_file: the path to the results file
        @return: None
        """
        df = pd.read_csv(results_file)
        st.write(df)

    def load_results_preds(self, results_file: Path) -> Tuple[List[str], List[str]]:
        """
        Load the results from the json file.
        @return: list of results
        """
        with open(results_file, "r") as f:
            json_data = json.load(f)
        results = json_data['results']['train']
        instances = [result['Instance'] for result in results]
        preds = [result['Result'] for result in results]
        return instances, preds

    def display_sample_examples(self, results_file_name: str) -> None:
        """
        Display sample examples from the results file.
        @param results_file_name: the name of the results file
        @return: None
        """
        # select experiment file
        results_files = [f for f in results_folder.iterdir() if f.is_file() and f.name.startswith("experiment") and
                         results_file_name in f.name]
        results_names_to_display = {f.name.split("experiment_")[1].split('.json')[0]: f for f in results_files}
        results_file = st.sidebar.selectbox("Select experiment file", sorted(results_names_to_display.keys()))
        instances, preds = self.load_results_preds(results_names_to_display[results_file])
        st.write("Sample examples")
        for i in range(5):
            st.markdown(f"Instance: {instances[i]}".replace("\n", "\n\n"))
            st.write(f"Prediction: {preds[i]}")
            st.write("----")


if __name__ == "__main__":
    vr = VisualizeResults()
    vr.display_page()
