import numpy as np
import json
import sys
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants

RESULTS_FOLDER = ExperimentConstants.RESULTS_PATH


class HistogramOfSamples:
    def display_page(self):
        st.title("Histogram of Samples")
        datasets_folders_names = [f for f in RESULTS_FOLDER.iterdir() if f.is_dir()]
        datasets_names_to_display = {f.name: f for f in datasets_folders_names}
        dataset_file_name = st.sidebar.selectbox("Select dataset to visualize", list(datasets_names_to_display.keys()))
        selected_dataset_file_name = datasets_names_to_display[dataset_file_name]

        shot_folders_name = [f for f in selected_dataset_file_name.iterdir() if f.is_dir()]
        shot_folders_name = {f.name: f for f in shot_folders_name}
        shot_file_name = st.sidebar.selectbox("Select the number of shots to visualize", list(shot_folders_name.keys()))
        selected_shot_file_name = shot_folders_name[shot_file_name]

        # find the csv file in the folder if exists
        result_files = [f for f in selected_shot_file_name.iterdir() if f.is_file() and f.name.endswith(".csv")]
        if result_files:
            train_file = [f for f in result_files if "train" in f.name and "accuracy" in f.name]
            assert len(train_file) >= 1, f"More than one train file found in the folder {selected_shot_file_name}"
            test_file = [f for f in result_files if "test" in f.name and "accuracy" in f.name]
            assert len(test_file) >= 1, f"More than one test file found in the folder {selected_shot_file_name}"
            result_files_to_display = {}
            if len(train_file) == 1:
                result_files_to_display['train_examples'] = train_file[0]
            if len(test_file) == 1:
                result_files_to_display['test_examples'] = test_file[0]
            result_file_name = st.sidebar.selectbox("Select the results file to visualize", result_files_to_display)
            result_file = result_files_to_display[result_file_name]
            df = self.display_samples(result_file)
            self.plot_histogram(df)
            self.display_sample_examples(selected_shot_file_name, dataset_file_name)
        else:
            st.markdown("No results file found in the folder")
            st.stop()

    def display_samples(self, results_file: Path):
        """
        Display the results of the model.

        @param results_file: the path to the results file
        @return: None
        """
        df = pd.read_csv(results_file)
        # sum each row to get the total number of instances (sum the ones in the row and divide by the number of
        # ones + zeros)
        predictions_columns = [col for col in df.columns if "experiment_template" in col]
        df['count_true_preds'] = df[predictions_columns].sum(axis=1)
        df['num_of_instances'] = df[predictions_columns].notnull().sum(axis=1)
        # count the values for each row
        df['accuracy'] = df['count_true_preds'] / df['num_of_instances']
        # multiply the accuracy by 100
        df['accuracy'] = round(df['accuracy'] * 100, 2)
        # put the accuracy in the first column
        df = df[['num_of_instances', 'accuracy']+predictions_columns]
        # add name to the index column
        df.index.name = 'example number'
        st.write(df)
        return df

    def load_results_preds_gt(self, results_file: Path) -> Tuple[List[str], List[str], List[str]]:
        """
        Load the results from the json file.
        @return: list of results
        """
        with open(results_file, "r") as f:
            json_data = json.load(f)
        results = json_data['results']['train']
        instances = [result['Instance'] for result in results]
        preds = [result['Result'] for result in results]
        gt = [result['GroundTruth'] for result in results]
        return instances, preds, gt

    def display_sample_examples(self, results_folder: Path, dataset_file_name: str) -> None:
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
        instances, preds, gt = self.load_results_preds_gt(datasets_names_to_display[results_file])
        st.write("Sample examples")
        if "file_index" not in st.session_state:
            st.session_state["file_index"] = 0
            st.session_state["files_number"] = len(instances)

        # add bottoms to choose example
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button(label="Previous sentence for tagging", on_click=ExamplesNavigator.previous_sentence)
        with col2:
            st.button(label="Next sentence for tagging", on_click=ExamplesNavigator.next_sentence)
        st.sidebar.selectbox(
            "Sentences",
            [f"sentence {i}" for i in range(0, st.session_state["files_number"])],
            index=st.session_state["file_index"],
            on_change=ExamplesNavigator.go_to_sentence,
            key="selected_sentence",
        )

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
    def plot_histogram(self, df):
        """
        Plot the histogram of the results.
        @param df: DataFrame containing the data to plot.
        """
        # Setting up the plot
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.grid(True)  # Add grid lines for better readability
        ax.set_axisbelow(True)  # Ensure grid lines are behind other plot elements

        # Plotting the histogram
        bins = np.arange(0, 105, 5)  # Adjust bins to include the range from 12 to 100
        df['accuracy'].plot(kind='hist', bins=bins, ax=ax, color='skyblue', edgecolor='black')

        # Adding title and labels
        ax.set_title("Histogram of Prediction Accuracy", fontsize=16)
        ax.set_xlabel("Number of Templates with Correct Prediction", fontsize=14)
        ax.set_ylabel("Number of Examples", fontsize=14)

        # Customizing tick labels from and plot all the bins
        ax.set_xticks(bins)
        ax.set_xticklabels([f'{i}' for i in bins])
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add the number of examples in the histogram bars in the center
        for rect in ax.patches:
            height = rect.get_height()
            if height:
                ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12, color='blue', fontweight='bold')

        # Display the plot
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        # plt.show()
        st.pyplot(fig)

def plot_histogram2(self, df):
        """
        Plot the histogram of the results.
        @param df:
        @return:
        """
        # Plotting the histogram of the accuracy from 0 to 100

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True)  # Add grid lines for better readability
        ax.set_axisbelow(True)  # Ensure grid lines are behind other plot elements

        df['accuracy'].plot(kind='hist', bins=20, ax=ax, color='skyblue', edgecolor='black')

        # Adding title and labels
        ax.set_title("Histogram of Prediction Accuracy", fontsize=16)
        ax.set_xlabel("Number of Templates with Correct Prediction", fontsize=14)
        ax.set_ylabel("Number of Examples", fontsize=14)

        # Customizing tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        # add the number of examples in the histogram bars in the center
        for i in ax.patches:
            ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.5, str(int(i.get_height())), ha='center',
                    va='bottom', fontsize=10)
        # Display the plot
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        st.pyplot(fig)

class ExamplesNavigator:
    @staticmethod
    def next_sentence():
        file_index = st.session_state["file_index"]
        if file_index < st.session_state["files_number"] - 1:
            st.session_state["file_index"] += 1

        else:
            st.warning('This is the last sentence.')

    @staticmethod
    def previous_sentence():
        file_index = st.session_state["file_index"]
        if file_index > 0:
            st.session_state["file_index"] -= 1
        else:
            st.warning('This is the first sentence.')


    @staticmethod
    def go_to_sentence():
        # split the number of the sentence from the string of st.session_state["sentence_for_tagging"]
        # and then convert it to int
        sentence_number = int(st.session_state["selected_sentence"].split(" ")[1]) - 1
        st.session_state["file_index"] = sentence_number

if __name__ == '__main__':
    hos = HistogramOfSamples()
    hos.display_page()
