import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.streamlit_app.ui_components.SamplesNavigator import SamplesNavigator
from src.utils.Constants import Constants

class MetaHistogramOfSamples:
    def __init__(self):
        self.constants = Constants()
        self.result_constants = self.constants.ResultConstants
        self.mmlu_constants = self.constants.MMLUConstants
        self.experiment_constants = self.constants.ExperimentConstants

    def display_page(self):
        st.title("Histogram of Samples")
        files_info = self.get_files()
        self.display_aggregated_results(*files_info)

    def get_files(self):
        main_results_path = self.experiment_constants.MAIN_RESULTS_PATH
        selected_results_file = ResultsLoader.get_folder_selection_options(
            main_results_path, "Select results folder to visualize", reverse=True
        )
        shot = st.sidebar.selectbox("Select number of shots", ["zero_shot", "three_shot"])
        model_files = self.get_model_files(selected_results_file)
        return selected_results_file, model_files, shot

    def get_model_files(self, selected_results_file):
        folders = [file for file in selected_results_file.iterdir() if file.is_dir()]
        sorted_folders = sorted((f.name, f) for f in folders)
        models = st.sidebar.multiselect("Select models to visualize", sorted_folders)
        return [f for _, f in models]

    def display_aggregated_results(self, results_file, models_files, shot):
        split_option = st.selectbox("Aggregate the dataset by:", self.mmlu_constants.SPLIT_OPTIONS)
        data_options = MMLUSplitter.get_data_options(split_option)
        split_option_value = st.selectbox("Select the split option value:", data_options)
        datasets_names = MMLUSplitter.get_data_files(split_option, split_option_value)

        total_merge_df = self.aggregate_data(results_file, models_files, shot, datasets_names)
        self.plot_aggregated_histogram(total_merge_df, split_option, split_option_value)
        self.display_examples(total_merge_df)

    def aggregate_data(self, results_file, models_files, shot, datasets_names):
        total_merge_df = pd.DataFrame()
        shot_suffix = Path(shot) / "empty_system_format"

        for model_file in models_files:
            merged_df = pd.DataFrame()
            for dataset_name in datasets_names:
                file_path = results_file / model_file / dataset_name / shot_suffix / "comparison_matrix_test_data.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df = self.process_data_frame(df, model_file.name)
                    merged_df = pd.concat([merged_df, df], axis=0)

            total_merge_df = pd.concat([total_merge_df, merged_df], axis=1)

        self.calculate_accuracy(total_merge_df)
        return total_merge_df

    def process_data_frame(self, df, model_name):
        df['accuracy'] = df.apply(lambda row: self.calculate_row_accuracy(row), axis=1)
        df.columns = [f"{model_name}_{col}" for col in df.columns]
        return df

    def calculate_row_accuracy(self, row):
        correct_predictions = row.filter(like='correct').sum()
        total_predictions = row.filter(like='total').count()
        return 100 * (correct_predictions / total_predictions)

    def calculate_accuracy(self, df):
        df['correct'] = df.sum(axis=1)
        df['number_of_predictions'] = df.count(axis=1)
        df['accuracy'] = df['correct'] / df['number_of_predictions'] * 100

    def plot_aggregated_histogram(self, df, split_option, split_option_value):
        title = f"Aggregated Histogram by {split_option} {split_option_value}"
        self.plot_histogram(df, title)

    def plot_histogram(self, df, title):
        fig, ax = plt.subplots()
        df['accuracy'].plot(kind='hist', bins=np.arange(0, 105, 5), ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    def display_examples(self, df):
        min_percentage = st.slider("Minimum percentage of examples to display", 0, 100, 0, step=5)
        max_percentage = st.slider("Maximum percentage of examples to display", 0, 100, 5, step=5)
        examples = df[(df['accuracy'] > min_percentage) & (df['accuracy'] < max_percentage)]
        self.display_example_details(examples)

    def display_example_details(self, examples):
        # Implementation to display details for each example
        pass

if __name__ == '__main__':
    hos = MetaHistogramOfSamples()
    hos.display_page()
