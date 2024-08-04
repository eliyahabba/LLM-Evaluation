import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants

ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
from src.streamlit_app.ui_components.SamplesNavigator import SamplesNavigator


class SaveSamples:

    def run(self):
        results_folder, models, shot = self.get_files()
        self.display_aggregated_results(results_folder, models, shot)

    def get_model_files(self, selected_results_file):
        folders = [file for file in selected_results_file.iterdir() if file.is_dir()]
        models_names = {f.name: f for f in folders}
        sorted_folders = dict(sorted(models_names.items(), key=lambda x: (x[0].lower(), x[0]), reverse=False))
        models = st.sidebar.multiselect("Select models to visualize", list(sorted_folders.keys()),
                                        default=list(sorted_folders.keys()))
        selected_models_files = [models_names[model] for model in models]
        return selected_models_files

    def get_files(self):
        results_folder = ResultConstants.MAIN_RESULTS_PATH / Path("MultipleChoiceTemplatesStructured")
        shot = "zero_shot"
        models = ["Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat-hf", "OLMo-1.7-7B-hf"]
        models_files = [results_folder / model for model in models]
        return results_folder, models_files, shot

    def display_aggregated_results(self, selected_results_file, selected_models_files, shot):
        data_options = MMLUSplitter.get_data_options(MMLUConstants.ALL_NAMES)
        datasets_names = MMLUSplitter.get_data_files(MMLUConstants.ALL_NAMES, data_options)
        total_merge_df = MetaHistogramCalculator.aggregate_data_across_models(
            selected_results_file,
                                                                              selected_models_files, shot,
                                                                              datasets_names)
        total_merge_df = MetaHistogramCalculator.calculate_and_add_accuracy_columns(total_merge_df)
        self.get_examples(selected_models_files, total_merge_df)

    def get_examples(self, models, df, min_percentage = 0, max_percentage = 5):
        examples = df[(df['accuracy'] > min_percentage) & (df['accuracy'] < max_percentage)]
        example_data = MetaHistogramCalculator.extract_example_data(examples)
        self.create_text_examples(example_data, models)

    def create_text_examples(self, example_data, models):
        sample_cols = ["Dataset" , "Index", "Instance",  "GroundTruth",  "Result", "Score"]
        import pandas as pd
        sample_df = pd.DataFrame(columns=['model']+sample_cols)
        for model in models:
            for dataset in  example_data['dataset'].unique():
                results_folder = ResultConstants.MAIN_RESULTS_PATH / Path("MultipleChoiceTemplatesStructured")
                selected_examples = example_data[example_data['dataset'] == dataset]
                # current_instance = selected_examples.iloc[st.session_state.get('file_index', 0)]
                for current_instance in range(len(selected_examples)):
                    full_results_path = results_folder / model / dataset / "zero_shot" / "empty_system_format" / "experiment_template_0.json"
                    with open(full_results_path, "r") as file:
                        template = json.load(file)
                    sample = template["results"]["test"][current_instance]
                    row = [model, dataset, sample["Index"], sample["Instance"], sample["GroundTruth"], sample["Result"], sample["Score"]]
        # create a df from the samples
                    sample_df = pd.concat([sample_df, pd.DataFrame(row, columns=sample_cols)])
        # sort by dataset, index
        sample_df = sample_df.sort_values(by=["Dataset", "Index"])
        return sample_df



if __name__ == '__main__':
    hos = SaveSamples()
    hos.run()
