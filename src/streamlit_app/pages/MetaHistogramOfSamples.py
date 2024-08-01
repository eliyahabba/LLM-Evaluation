import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  # For creating the bar plot
import streamlit as st

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.streamlit_app.ui_components.SamplesNavigator import SamplesNavigator
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants

MMLUConstants = Constants.MMLUConstants
ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class MetaHistogramOfSamples:
    def get_files(self):
        selected_results_file = ResultsLoader.get_folder_selections_options(MAIN_RESULTS_PATH,
                                                                            "Select results folder to visualize",
                                                                            reverse=True)
        shot = st.sidebar.selectbox("select number of shots", ["zero_shot", "three_shot"])

        # select all the possible models:
        folders_names = [file for file in selected_results_file.iterdir() if file.is_dir()]
        names_to_display = {f.name: f for f in folders_names}
        names_to_display = dict(sorted(names_to_display.items(), key=lambda x: (x[0].lower(), x[0]), reverse=False))
        # multi select the models
        text_to_display = "Select models to visualize"
        models = st.sidebar.multiselect(text_to_display, list(names_to_display.keys()),
                                        default=list(names_to_display.keys()))
        # models = st.sidebar.multiselect(text_to_display, list(names_to_display.keys()), default=[])
        selected_models_files = [names_to_display[model] for model in models]

        return selected_results_file, selected_models_files, shot

    def display_page(self):
        st.title("Histogram of Samples")
        selected_results_file, selected_models_files, shot = self.get_files()

        self.new_display_aggregated_results(selected_results_file, selected_models_files, shot)

    def new_display_aggregated_results(self, selected_results_file, selected_models_files, shot):
        split_option = st.selectbox("aggregated the dataset by:", MMLUConstants.SPLIT_OPTIONS,
                                    MMLUConstants.SPLIT_OPTIONS.index(MMLUConstants.ALL_NAMES))

        data_options = MMLUSplitter.get_data_options(split_option)
        split_option_value = st.selectbox("select the split option value:", data_options)
        datasets_names = MMLUSplitter.get_data_files(split_option, split_option_value)
        shot_suffix = Path(shot) / Path("empty_system_format")
        total_merge_df = pd.DataFrame()
        for model_file in selected_models_files:
            mmlu_files = [selected_results_file / model_file / Path(datasets_name) / shot_suffix
                          / Path("comparison_matrix_test_data.csv") for datasets_name in
                          datasets_names]
            # filter non existing files
            mmlu_files = [file for file in mmlu_files if file.exists()]
            if len(mmlu_files) > 0:
                merged_df = pd.DataFrame()
                for mmlu_file in mmlu_files:
                    df = self.display_samples_prediction_accuracy(mmlu_file, display_results=False)
                    # reset the index and add prfix mmlu_file.parents[2].name + the index of the row
                    df = df.reset_index()
                    df['example_number'] = mmlu_file.parents[2].name + "_" + df.index.astype(str)
                    merged_df = pd.concat([merged_df, df])
                # concat on the example_number column such that sum the values of the same example_number to each cell
                # reomove the col accuarcy and num_of_predictions
                merged_df = merged_df.drop(columns=['accuracy', 'num_of_predictions'])
                # set the index to example_number
                merged_df = merged_df.set_index('example_number')
                # add to each column the name of the model
                merged_df.columns = [model_file.name + "_" + col for col in merged_df.columns]
                total_merge_df = pd.concat([total_merge_df, merged_df], axis=1)
        # total_merge_df = total_merge_df.groupby('example_number').sum().reset_index()
        # ad accuracy column and the values is the sum of this row
        total_merge_df['correct'] = total_merge_df.sum(axis=1, skipna=True)
        total_merge_df['number_of_predictions'] = total_merge_df.count(
            axis=1) - 1  # Subtract 1 to exclude the 'accuracy' column itself
        total_merge_df['accuracy'] = (total_merge_df['correct'] / total_merge_df['number_of_predictions']) * 100
        # add the accuracy column
        self.plot_aggregated_histogram(total_merge_df, split_option, split_option_value)
        self.display_examples(total_merge_df)

    def display_examples(self, total_merge_df):
        pass
        min_percatnge = st.slider("select the minimum percentage of examples to display", 0, 100, 0, step=5)
        max_percatnge = st.slider("select the maximum percentage of examples to display", 0, 100, 5, step=5)
        # take all the valu that the accuracy is lower than the percentage
        examples = total_merge_df[
            (total_merge_df['accuracy'] > min_percatnge) & (total_merge_df['accuracy'] < max_percatnge)]
        # now split the examples to the dataset name and the example_number (when start with digits)
        examples_id = examples.index.to_series()
        dataset_index = examples_id.str.rsplit('_', n=1, expand=True)
        # create a new df from tha accuarcy column and dataset and index
        examples_id = pd.DataFrame(
            {'dataset': dataset_index[0], 'example_number': dataset_index[1], 'accuracy': examples['accuracy']})

        # display bar plot of the number of examples per dataset
        fig = px.bar(examples_id['dataset'].value_counts(), x=examples_id['dataset'].value_counts().index,
                     y=examples_id['dataset'].value_counts().values, labels={'x': 'dataset', 'y': 'number of examples'},
                     title='Number of examples for each dataset in the selected range')
        # add the number of each bar on the top of the bar
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        # rotate the x axis labels by 45 degrees
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
        # display the examples
        self.display_text_examples(examples_id)

    def display_samples_prediction_accuracy(self, results_file: Path, display_results=True):
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
        df['num_of_predictions'] = df[predictions_columns].notnull().sum(axis=1)
        # count the values for each row
        df['accuracy'] = df['count_true_preds'] / df['num_of_predictions']
        # multiply the accuracy by 100
        df['accuracy'] = round(df['accuracy'] * 100, 2)
        # put the accuracy in the first column
        df = df[['num_of_predictions', 'accuracy'] + predictions_columns]
        # add name to the index column
        df.index.name = 'example_number'
        if display_results:
            st.write(df)
        return df

    def plot_histogram(self, df, title="Histogram of Prediction Accuracy"):
        """
        Plot the histogram of the results.
        @param df: DataFrame containing the data to plot.
        """
        st.markdown(f"There are {len(df)} examples in the dataset.")
        # Setting up the plot
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.grid(True)  # Add grid lines for better readability
        ax.set_axisbelow(True)  # Ensure grid lines are behind other plot elements

        # Plotting the histogram
        bins = np.arange(0, 105, 5)  # Adjust bins to include the range from 12 to 100
        df['accuracy'].plot(kind='hist', bins=bins, ax=ax, color='skyblue', edgecolor='black')

        # Adding title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Percentage of Templates with Correct Predictions", fontsize=14)
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

    def plot_aggregated_histogram(self, merged_df, split_option, split_option_value):
        value = f"on {split_option_value}" if split_option_value else ""
        self.plot_histogram(merged_df, title=f"Aggregated Histogram by {split_option} {value}")

    def display_text_examples(self, examples_id):
        """
        Display sample examples from the results file.
        @param results_folder: the path to the results folder
        @return: None
        """
        # select experiment file to display
        results_folder = ResultConstants.MAIN_RESULTS_PATH / Path("MultipleChoiceTemplatesStructured")
        default_model = "Mistral-7B-Instruct-v0.2"
        # ask the use to select the model
        models = sorted([model for model in os.listdir(results_folder) if os.path.isdir(results_folder / model)])
        model = st.selectbox("Select model", models,
                             index=models.index(default_model) if default_model in models else 0)

        shot = "zero_shot"
        system_format = "empty_system_format"

        datasets = examples_id.dataset.value_counts().index
        # ask the user to select the dataset
        dataset = st.selectbox("Select dataset", datasets)
        # get the examples for the selected dataset
        dataset_index = examples_id[examples_id['dataset'] == dataset]['example_number'].values

        # sort the files by the number of the experiment
        # write on the center of the page
        st.markdown(f"#### {len(dataset_index)} Examples", unsafe_allow_html=True)
        if "file_index" not in st.session_state:
            st.session_state["file_index"] = 0
        if "dataset" not in st.session_state:
            st.session_state["dataset"] = dataset
        if dataset != st.session_state["dataset"]:
            st.session_state["file_index"] = 0
            st.session_state["dataset"] = dataset

        st.session_state["files_number"] = len(dataset_index)

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

        current_instance = examples_id[examples_id['example_number'] == dataset_index[st.session_state['file_index']]]
        current_instance = int(current_instance.example_number.values[0])
        full_results_path = results_folder / Path(model) / Path(dataset) / Path(shot) / Path(system_format) / Path(
            "experiment_template_0.json")
        with open(full_results_path, "r") as f:
            template = json.load(f)
        sample = template["results"]["test"][current_instance]
        formatted_str = sample['Instance'].replace("\n\n", "<br><br>").replace("\n", "<br>")
        st.markdown(f"**Instance**: {formatted_str}", unsafe_allow_html=True)
        st.markdown(f"**Ground True**: {sample['GroundTruth']}")
        st.markdown(f"**Predicted**: {sample['Result']}")
        st.markdown(f"**Score**: {sample['Score']}")
        st.write("----")


if __name__ == '__main__':
    hos = MetaHistogramOfSamples()
    hos.display_page()
