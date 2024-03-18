import json
import sys
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import streamlit as st

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants

RESULTS_FOLDER = ExperimentConstants.RESULTS_PATH


class VisualizeResults:
    def display_page(self):
        st.title("Templates Visualization")

        datasets_folders_names = [f for f in RESULTS_FOLDER.iterdir() if f.is_dir()]
        datasets_names_to_display = {f.name: f for f in datasets_folders_names}
        dataset_file_name = st.sidebar.selectbox("Select dataset to visualize", list(datasets_names_to_display.keys()))
        selected_dataset_file_name = datasets_names_to_display[dataset_file_name]

        self.display_possible_templates_args()

        shot_folders_name = [f for f in selected_dataset_file_name.iterdir() if f.is_dir()]
        shot_folders_name = {f.name: f for f in shot_folders_name}
        shot_file_name = st.sidebar.selectbox("Select the number of shots to visualize", list(shot_folders_name.keys()))
        selected_shot_file_name = shot_folders_name[shot_file_name]

        # find the csv file in the folder if exists
        result_files = [f for f in selected_shot_file_name.iterdir() if f.is_file() and f.name.endswith(".csv")]
        if result_files:
            train_file = [f for f in result_files if "train" in f.name and "scores" in f.name]
            assert len(train_file) >= 1, f"More than one train file found in the folder {selected_shot_file_name}"
            test_file = [f for f in result_files if "test" in f.name and "scores" in f.name]
            assert len(test_file) >= 1, f"More than one test file found in the folder {selected_shot_file_name}"
            result_files_to_display = {}
            if len(train_file) == 1:
                result_files_to_display['train_examples'] = train_file[0]
            if len(test_file) == 1:
                result_files_to_display['test_examples'] = test_file[0]
            result_file_name = st.sidebar.selectbox("Select the results file to visualize", result_files_to_display)
            result_file = result_files_to_display[result_file_name]
            self.display_results(result_file)
            dataset_type = result_file_name.split("_")[0]
            self.display_sample_examples(selected_shot_file_name, dataset_file_name, dataset_type)
        else:
            st.markdown("No results file found in the folder")
            st.stop()

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

    def load_results_preds_gt(self, results_file: Path, dataset_type:str) -> Tuple[List[str], List[str], List[str]]:
        """
        Load the results from the json file.
        @return: list of results
        """
        with open(results_file, "r") as f:
            json_data = json.load(f)
        results = json_data['results'][dataset_type]
        instances = [result['Instance'] for result in results]
        preds = [result['Result'] for result in results]
        gt = [result['GroundTruth'] for result in results]
        return instances, preds, gt

    def display_sample_examples(self, results_folder: Path, dataset_file_name: str, dataset_type:str) -> None:
        """
        Display sample examples from the results file.
        @param results_folder: the path to the results folder
        @return: None
        """
        # select experiment file
        datasets_names_to_display = {f.name.split("experiment_")[1].split('.json')[0]: f for f in
                                     results_folder.iterdir() if
                                     f.is_file() and f.name.endswith(".json")}
        # sort the files by the number of the experiment
        datasets_names_to_display = dict(
            sorted(datasets_names_to_display.items(), key=lambda item: int(item[0].split("_")[1])))
        results_file = st.sidebar.selectbox("Select template file", list(datasets_names_to_display.keys()))
        instances, preds, gt = self.load_results_preds_gt(datasets_names_to_display[results_file], dataset_type)
        st.write("Sample examples")
        for i in range(3):
            current_instance = instances[st.session_state["file_index"]]
            formatted_str = current_instance.replace("\n\n", "<br><br>").replace("\n", "<br>")
            st.markdown(f"**Instance**: {formatted_str}", unsafe_allow_html=True)
            st.write(f"**Prediction**: {preds[st.session_state['file_index']]}")
            st.write(f"**Ground True**: {gt[st.session_state['file_index']]}")
            st.write("----")

        self.load_template(results_file, dataset_file_name)

    def load_template(self, results_file, dataset_file_name):
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        template_path = templates_path / dataset_file_name / Path(f"{results_file}.json")
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
