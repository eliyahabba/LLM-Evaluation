import json
from pathlib import Path

import streamlit as st

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants

ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class SaveSamples:
    def __init__(self, models, shot):
        self.models = models
        self.shot = shot

    def run(self):
        results_folder, models = self.get_files()
        agg_df = self.get_aggregated_results(results_folder, models)
        self.save_samples(agg_df)

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
        models_files = [results_folder / model for model in self.models]
        return results_folder, models_files

    def get_aggregated_results(self, selected_results_file, selected_models_files):
        data_options = MMLUSplitter.get_data_options(MMLUConstants.ALL_NAMES)
        datasets_names = MMLUSplitter.get_data_files(MMLUConstants.ALL_NAMES, data_options)
        total_merge_df = MetaHistogramCalculator.aggregate_data_across_models(selected_results_file,
                                                                              selected_models_files, self.shot,
                                                                              datasets_names)
        total_merge_df = MetaHistogramCalculator.calculate_and_add_accuracy_columns(total_merge_df)
        agg_df = self.get_examples(selected_models_files, total_merge_df)
        return agg_df

    def get_examples(self, models, df, min_percentage=0, max_percentage=5):
        examples = df[(df['accuracy'] > min_percentage) & (df['accuracy'] < max_percentage)]
        example_data = MetaHistogramCalculator.extract_example_data(examples)
        agg_df = self.create_text_examples(example_data, models)
        return agg_df

    def create_text_examples(self, example_data, models):
        sample_cols = ["Dataset", "Index", "Instance", "GroundTruth", "Result", "Score"]
        import pandas as pd
        sample_df = pd.DataFrame(columns=['model'] + sample_cols)
        for model in models:
            for dataset in example_data['dataset'].unique():
                results_folder = ResultConstants.MAIN_RESULTS_PATH / Path("MultipleChoiceTemplatesStructured")
                selected_examples = example_data[example_data['dataset'] == dataset]
                for current_instance in range(len(selected_examples)):
                    full_results_path = results_folder / model / dataset / "zero_shot" / "empty_system_format" / "experiment_template_0.json"
                    with open(full_results_path, "r") as file:
                        template = json.load(file)
                    sample = template["results"]["test"][current_instance]
                    row_dict = {
                        "model": model.name,
                        "Dataset": dataset,
                        "Index": sample["Index"],
                        "Instance": sample["Instance"],
                        "GroundTruth": sample["GroundTruth"],
                        "Result": sample["Result"],
                        "Score": sample["Score"]
                    }
                    row_df = pd.DataFrame([row_dict], columns=sample_df.columns)

                    # Concatenate the new row to the existing DataFrame
                    sample_df = pd.concat([sample_df, row_df], ignore_index=True)
        # create a df from the samples
        # sort by dataset, index
        sample_df = sample_df.sort_values(by=["Dataset", "Index"])
        return sample_df

    def save_samples(self, agg_df):
        agg_df.to_csv("samples.csv", index=False)
        pass


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    models = ["Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat-hf", "OLMo-1.7-7B-hf"]
    args.add_argument("--models", type=list, help="List of models to visualize", default=models)
    args.add_argument("--shot", type=str, help="Shot type", default="zero_shot", choices=["zero_shot", "three_shot"])
    args = args.parse_args()
    save_samples = SaveSamples(models=args.models, shot=args.shot)
    save_samples.run()
