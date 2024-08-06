import argparse
import json

import pandas as pd

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants
SamplesAnalysisConstants = Constants.SamplesAnalysisConstants
ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class SampleSaver:
    def __init__(self, models_to_save, shot, output_path):
        self.models_to_save = models_to_save
        self.shot = shot
        self.output_path = output_path

    def execute(self):
        results_folder, models_files = self.retrieve_files()
        aggregated_df = self.aggregate_results(results_folder, models_files)
        self.save_aggregated_samples(aggregated_df)

    def retrieve_files(self):
        self.results_folder = ResultConstants.MAIN_RESULTS_PATH / "MultipleChoiceTemplatesStructured"
        model_files = [file for file in self.results_folder.iterdir() if file.is_dir()]
        # models_names = {f.name: f for f in folders}
        # selected_models_files = [models_names[model] for model in models]
        # return selected_models_files
        #
        # model_files = [results_folder / model for model in self.models_to_save]
        return self.results_folder, model_files

    def aggregate_results(self, results_folder, model_files):
        data_options = MMLUSplitter.get_data_options(MMLUConstants.ALL_NAMES)
        dataset_names = MMLUSplitter.get_data_files(MMLUConstants.ALL_NAMES, data_options)
        total_aggregated_df = MetaHistogramCalculator.aggregate_data_across_models(results_folder, model_files,
                                                                                   self.shot, dataset_names)
        total_aggregated_df = MetaHistogramCalculator.calculate_and_add_accuracy_columns(total_aggregated_df)
        # sort bu index
        total_aggregated_df = total_aggregated_df.sort_index()
        aggregated_df = self.fetch_examples(total_aggregated_df)
        return aggregated_df

    def fetch_examples(self, dataframe, min_accuracy=0, max_accuracy=5):
        example_rows = dataframe[(dataframe['accuracy'] > min_accuracy) & (dataframe['accuracy'] < max_accuracy)]
        example_data = MetaHistogramCalculator.extract_example_data(example_rows)
        aggregated_df = self.compose_text_examples(example_data)
        return aggregated_df

    def compose_text_examples(self, example_data):
        sample_columns = ["Dataset", "Index", "Instance", "GroundTruth", "Result", "Score"]
        samples_df = pd.DataFrame(columns=['model'] + sample_columns)
        for model in self.models_to_save:
            for dataset in example_data['dataset'].unique():
                dataset_examples = example_data[example_data['dataset'] == dataset]['example_number'].values
                for index in dataset_examples:
                    result_path = self.results_folder / model / dataset / "zero_shot" / "empty_system_format" / "experiment_template_0.json"
                    with open(result_path, "r") as file:
                        template = json.load(file)
                    result = template["results"]["test"][int(index)]
                    row = {
                        "model": model,
                        "Dataset": dataset,
                        "Index": result["Index"],
                        "Instance": result["Instance"],
                        "GroundTruth": result["GroundTruth"],
                        "Result": result["Result"],
                        "Score": int(result["Score"])
                    }
                    row_df = pd.DataFrame([row], columns=samples_df.columns)
                    samples_df = pd.concat([samples_df, row_df], ignore_index=True)
        samples_df = samples_df.sort_values(by=["Dataset", "Index"])
        samples_df["Score"] = samples_df["Score"].astype(int)
        return samples_df

    def save_aggregated_samples(self, aggregated_df):
        aggregated_df.to_csv(self.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    models_default = ["Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat-hf", "OLMo-7B-Instruct-hf"]
    parser.add_argument("--models_to_save", type=list, default=models_default, help="List of models to visualize")
    parser.add_argument("--shot", type=str, default="zero_shot", choices=["zero_shot", "three_shot"], help="Shot type")
    parser.add_argument("--output_path", type=str, default=SamplesAnalysisConstants.SAMPLES_PATH, help="Path to save the samples")
    args = parser.parse_args()
    sample_saver = SampleSaver(models_to_save=args.models_to_save, shot=args.shot, output_path=args.output_path)
    sample_saver.execute()
