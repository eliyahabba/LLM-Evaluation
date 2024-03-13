import json
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import streamlit as st

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants

RESULTS_FOLDER = ExperimentConstants.RESULTS_PATH


class VisualizeResults:
    def display_page(self):
        st.title("Templates Visualization")

        results_files = [f for f in RESULTS_FOLDER.iterdir() if f.is_dir()]
        results_names_to_display = {f.name: f for f in results_files}

        card_file_name = st.sidebar.selectbox("Select dataset to visualize", list(results_names_to_display.keys()))
        results_folder = results_names_to_display[card_file_name]
        # find the csv file in the folder if exists
        result_file = [f for f in results_folder.iterdir() if f.is_file() and f.name.endswith(".csv")][0] if \
            [f for f in results_folder.iterdir() if f.is_file() and f.name.endswith(".csv")] else None
        if result_file:
            results_file = result_file
            self.display_results(results_file)
            self.display_sample_examples(results_folder)

        else:
            st.markdown("No results file found in the folder")

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

    def display_sample_examples(self, results_folder: Path) -> None:
        """
        Display sample examples from the results file.
        @param card_file_name: the name of the results file
        @return: None
        """
        # select experiment file
        results_names_to_display = {f.name.split("experiment_")[1].split('.json')[0]: f for f in
                                    results_folder.iterdir() if
                                    f.is_file() and f.name.endswith(".json")}
        # sort the files by the number of the experiment
        results_names_to_display = dict(sorted(results_names_to_display.items(), key=lambda item: int(item[0].split("_")[1])))
        results_file = st.sidebar.selectbox("Select template file", list(results_names_to_display.keys()))
        instances, preds = self.load_results_preds(results_names_to_display[results_file])
        st.write("Sample examples")
        for i in range(5):
            formatted_str = instances[i].replace("\n\n", "<br><br>").replace("\n", "<br>")
            # st.markdown(f"Instance: {instances[i]}".replace("\n", "\n\n"))
            st.markdown(f"Instance: {formatted_str}", unsafe_allow_html=True)
            # st.markdown(f"Instance: {instances[i]}")
            st.write(f"Prediction: {preds[i]}")
            st.write("----")

        self.load_template(results_folder, results_file)

    def load_template(self, results_folder, results_file):
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        card = results_folder.name
        template_path = templates_path / Path(card) / Path(f"{results_file}.json")
        with open(template_path, "r") as f:
            template = json.load(f)
        # template is a dict, print each key value pair in the sidebar
        for key, value in template.items():
            if value == "\n":
                value = value.replace('\n', '\\n')
                st.sidebar.markdown(f"{key}: {value}")
            elif value == " ":
                value = value.replace(' ', '\\s')
                st.sidebar.markdown(f"**{key}** : {value}")
            else:
                st.sidebar.markdown(f"**{key}** : {value}")



if __name__ == "__main__":
    vr = VisualizeResults()
    vr.display_page()
