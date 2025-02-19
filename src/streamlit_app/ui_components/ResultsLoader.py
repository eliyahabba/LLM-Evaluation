import json
from pathlib import Path
from typing import Tuple, List, Optional

import streamlit as st

from src.streamlit_app.ui_components.SamplesNavigator import SamplesNavigator
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
MMLUConstants = Constants.MMLUConstants

MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class FilesFilter:
    def __init__(self, filter_name: str):
        self.filter_name = filter_name

    def get_files_by_name(self, files: List[Path]) -> List[Path]:
        """
        Filter the datasets by the name.
        @param files: the files to filter
        @return: the filtered datasets
        """
        return [file for file in files if self.filter_name in file.name]


class ResultsLoader:
    @staticmethod
    def load_template(results_file, dataset_file_name):
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

    @staticmethod
    def load_results_preds_gt_scores(results_file: Path, dataset_type: str) -> Tuple[
        List[str], List[str], List[str], List[str]]:
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
        scores = [result['Score'] for result in results]
        return instances, preds, gt, scores

    @staticmethod
    def display_sample_examples(results_folder: Path, dataset_file_name: str, result_file_name: str) -> None:
        """
        Display sample examples from the results file.
        @param results_folder: the path to the results folder
        @return: None
        """
        # select experiment file
        dataset_type = result_file_name.split("_")[0]

        datasets_names_to_display = {f.name.split("experiment_")[1].split('.json')[0]: f for f in
                                     results_folder.iterdir() if
                                     f.is_file() and f.name.endswith(".json")}
        # sort the files by the number of the experiment
        datasets_names_to_display = dict(
            sorted(datasets_names_to_display.items(), key=lambda item: int(item[0].split("_")[1])))
        results_file = st.sidebar.selectbox("Select template file", list(datasets_names_to_display.keys()))
        instances, preds, gt, scores = ResultsLoader.load_results_preds_gt_scores(
            datasets_names_to_display[results_file],
            dataset_type)
        # write on the center of the page
        st.markdown(f"#### Examples: prompt + prediction", unsafe_allow_html=True)
        if "file_index" not in st.session_state:
            st.session_state["file_index"] = 0
        st.session_state["files_number"] = len(instances)

        # add bottoms to choose example
        col1, col2 = st.columns(2)
        with col1:
            st.button(label="Previous sentence", on_click=SamplesNavigator.previous_sentence)
        with col2:
            st.button(label="Next sentence", on_click=SamplesNavigator.next_sentence)
        st.selectbox(
            "Sentences",
            [f"sentence {i}" for i in range(0, st.session_state["files_number"])],
            index=st.session_state["file_index"],
            on_change=SamplesNavigator.go_to_sentence,
            key="selected_sentence",
        )

        current_instance = instances[st.session_state["file_index"]]
        formatted_str = current_instance.replace("\n\n", "<br><br>").replace("\n", "<br>")
        st.markdown(f"**Instance**: {formatted_str}", unsafe_allow_html=True)
        st.write(f"**Prediction**: {preds[st.session_state['file_index']]}")
        st.write(f"**Ground True**: {gt[st.session_state['file_index']]}")
        st.write(f"**Score**: {scores[st.session_state['file_index']]}")
        st.write("----")

        ResultsLoader.load_template(results_file, dataset_file_name)

    @staticmethod
    def get_folder_selections_options(folder_path: Path, text_to_display: str, reverse: bool = False,
                                      files_filter: FilesFilter = None) -> Path:
        """
        Get the folder selection options. The user can select the folder to visualize.
        @param folder_path: the path to the folder
        @param text_to_display: the text to display in the sidebar
        @param files_filter: the filter to apply on the files

        @return: the selected folder name
        """
        if isinstance(folder_path, list):
            results_file_name = st.sidebar.selectbox(text_to_display, list(folder_path))
            return results_file_name
        folders_names = [file for file in folder_path.iterdir() if file.is_dir()]
        if files_filter is not None:
            folders_names = files_filter.get_files_by_name(folders_names)
        names_to_display = {f.name: f for f in folders_names}
        # id Mistral in the name, the name of Mistral should be first
        names_to_display = dict(sorted(names_to_display.items(), key=lambda x: ("Mistral" not in x[0],
                                                                                "not_structured" in x[0],
                                                                                x[0].lower(), x[0])
                                       , reverse=reverse))
        results_file_name = st.sidebar.selectbox(text_to_display, list(names_to_display.keys()))
        selected_file_name = names_to_display[results_file_name]
        return selected_file_name

    @staticmethod
    def select_experiment_params():
        selected_results_file = ResultsLoader.get_folder_selections_options(MAIN_RESULTS_PATH,
                                                                            "Select results folder to visualize",
                                                                            reverse=True)
        selected_model_file = ResultsLoader.get_folder_selections_options(selected_results_file,
                                                                          "Select model to visualize")
        files_filter = FilesFilter("mmlu")
        # split_option = st.sidebar.selectbox("Split the dataset by:", MMLUConstants.SPLIT_OPTIONS)
        # #get the optinal dataset file
        # data_options = MMLUSplitter.get_data_options(split_option)
        #
        # selected_dataset_file = ResultsLoader.get_folder_selections_options(data_options,
        #                                                                     "Select dataset to visualize")
        # datasets_files = MMLUSplitter.get_data_files(split_option, selected_dataset_file)
        # select the first file to select the system format

        selected_dataset_file = ResultsLoader.get_folder_selections_options(selected_model_file,
                                                                            files_filter=files_filter,
                                                                            text_to_display="Select dataset to visualize")
        selected_shot_file = ResultsLoader.get_folder_selections_options(selected_dataset_file,
                                                                         "Select shot to visualize")
        selected_system_format = ResultsLoader.get_folder_selections_options(selected_shot_file,
                                                                             "Select system format to visualize")
        return selected_dataset_file.name, selected_system_format, selected_model_file

    @staticmethod
    def select_result_file(result_files, results_type_name="performance_summary"):
        train_file = [f for f in result_files if "train" in f.name and f.name.startswith(results_type_name)]
        assert len(train_file) <= 1, f"More than one train file found in the folder {train_file[0].parent}"
        test_file = [f for f in result_files if "test" in f.name and f.name.startswith(results_type_name)]
        assert len(test_file) <= 1, f"More than one test file found in the folder {test_file[0].parent}"
        result_files_to_display = {}
        if train_file:
            result_files_to_display['train_examples'] = train_file[0]
        if test_file:
            result_files_to_display['test_examples'] = test_file[0]
        result_file_name = st.sidebar.selectbox("Select the results file to visualize", result_files_to_display)
        result_file = result_files_to_display[result_file_name]
        return result_file_name, result_file

    @staticmethod
    def get_result_files(selected_shot_file_name: Path) -> Optional[List[Path]]:
        result_files = [f for f in selected_shot_file_name.iterdir() if f.is_file() and f.name.endswith(".csv")]
        if not result_files:
            ResultsLoader.display_no_results_message()
            return None
        else:
            return result_files

    @staticmethod
    def display_no_results_message() -> None:
        st.markdown("No results file found in the folder")
        st.stop()
