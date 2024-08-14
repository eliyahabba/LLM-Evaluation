import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants
SamplesAnalysisConstants = Constants.SamplesAnalysisConstants
ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class checkProbSamples:
    def __init__(self, models_to_save, shot, output_path, num_models, num_samples):
        self.models_to_save = models_to_save
        self.shot = shot
        self.output_path = output_path
        self.num_models = num_models
        self.num_samples = num_samples

    def execute(self):
        results_folder, models_files = self.retrieve_files()
        total_aggregated_df = self.aggregate_results(results_folder, models_files)
        self.add_bins(total_aggregated_df)
        scores = self.evaluate_models_across_bins(total_aggregated_df,
                                                  start_bin=20, end_bin=0,
                                                  num_models=self.num_models,
                                                  num_samples=self.num_samples)
        self.plot_heatmap(scores)
        self.plot_density(scores)
        self.plot_average_trend(scores)

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
        return total_aggregated_df

    def add_bins(self, total_aggregated_df):
        bins = list(range(0, 101, 5))
        labels = [f"{i}-{i + 5}" for i in range(0, 100, 5)]
        total_aggregated_df['accuracy_bin'] = pd.cut(total_aggregated_df['accuracy'], bins=bins, labels=labels, right=False)

    def evaluate_models_across_bins(self, df, start_bin, end_bin, num_models, num_samples=1000):
        """Evaluates model performance across bins."""
        scores = []

        current_label = f"{(start_bin-1) * 5}-{start_bin * 5}"
        current_bin_rows = df[df['accuracy_bin'] == current_label]
        num_samples = min(num_samples, len(current_bin_rows))
        sample_current_bin_rows = current_bin_rows.sample(n=num_samples)

        for _, row in sample_current_bin_rows.iterrows():
            row_score = [1]
            # Filter for models that scored 1 in the current example
            successful_models = [col for col in df.columns if 'template' in col and row[col] == 1]

            # Determine the number of models to sample
            sample_size = min(num_models, len(successful_models))

            if sample_size == 0:
                continue  # If no successful models, continue to next row

            # Sample models from the successful ones
            sampled_models = np.random.choice(successful_models, sample_size, replace=False)
            for current_bin in range(start_bin-1, end_bin, -1):
                # Convert bins to labels
                previous_label = f"{(current_bin - 1) * 5}-{current_bin * 5}"
                previous_bin_rows = df[df['accuracy_bin'] == previous_label]
            # Count successful models
                success_count = 0
                for model in sampled_models:
                    if row[model] == 1 and not previous_bin_rows[model].empty and previous_bin_rows[model].iloc[0] == 1:
                        success_count += 1

                # Calculate score for the current example
                # example_score = round(success_count / num_models, 2)
                row_score.append(success_count)

            # Calculate average score for the bin
            scores.append(row_score)

        return scores

    def plot_heatmap(self, scores):
        """Plot a heatmap of model success rates."""
        # Convert scores to a DataFrame
        score_matrix = pd.DataFrame(scores, columns=[f"{(i - 1) * 5}-{i * 5}" for i in range(20,   0, -1)])

        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Model Success Rates Across Bins')
        plt.xlabel('Accuracy Bins')
        plt.ylabel('Sample Index')
        plt.show()

    def plot_density(self, scores):
        """Plot density of model success rates for each bin transition."""
        # Convert scores to a DataFrame
        score_matrix = pd.DataFrame(scores, columns=[f"{(i - 1) * 5}-{i * 5}" for i in range(20, 0, -1)])

        plt.figure(figsize=(12, 8))
        for column in score_matrix.columns:
            sns.kdeplot(score_matrix[column], label=column, fill=True)
        plt.title('Density of Model Success Rates Across Bins')
        plt.xlabel('Success Rate')
        plt.ylabel('Density')
        plt.legend(title='Accuracy Bins')
        plt.show()

    def plot_average_trend(self, scores):
        """Plot average trend line of model success rates across bins."""
        # Convert scores to a DataFrame
        score_matrix = pd.DataFrame(scores, columns=[f"{(i - 1) * 5}-{i * 5}" for i in range(20, 0, -1)])

        average_scores = score_matrix.mean()

        plt.figure(figsize=(12, 8))
        plt.plot(average_scores.index, average_scores.values, marker='o', linestyle='-', color='b')
        plt.title('Average Model Success Rates Across Bins')
        plt.xlabel('Accuracy Bins')
        plt.ylabel('Average Success Rate')
        plt.grid(True)
        plt.show()

    # def sample_and_check_across_bins(self, df, start_bin, end_bin):
    #     """Samples rows across specified bins and checks model consistency across them."""
    #     # Add bins to the DataFrame
    #
    #     # Adjust the loop to go from high to low bins
    #     for current_bin in range(start_bin, end_bin - 1, -1):
    #         # Convert bins to labels
    #         current_label = f"{current_bin * 5}-{(current_bin + 1) * 5}"
    #         previous_label = f"{(current_bin - 1) * 5}-{current_bin * 5}"
    #
    #         current_bin_rows = df[df['accuracy_bin'] == current_label]
    #         previous_bin_rows = df[df['accuracy_bin'] == previous_label]
    #
    #         if current_bin_rows.empty or previous_bin_rows.empty:
    #             print(f"No sufficient data to compare bins {current_label} and {previous_label}.")
    #             continue
    #
    #         # Sample a row from the current bin
    #         sampled_row = current_bin_rows.sample(n=1)
    #
    #         # Find models that scored 1 in this row
    #         correct_models = sampled_row.columns[sampled_row.iloc[0] == 1]
    #         # Filter to keep only model columns
    #         correct_models = [model for model in correct_models if 'template' in model]
    #
    #         if not correct_models:
    #             print(f"No models scored 1 in the bin {current_label}.")
    #             continue
    #
    #         # Randomly select one of the models that scored 1
    #         model = np.random.choice(correct_models)
    #
    #         # Check this model in the rows of the previous bin
    #         previous_bin_correct = previous_bin_rows[model].sum()
    #
    #         print(
    #             f"From bin {current_label} to {previous_label}: Model {model} had {previous_bin_correct} correct out of {len(previous_bin_rows)} trials.")
    #
    #         # Optional: Add your criteria or further processing based on 'previous_bin_correct'
    # def plot_scores(self, scores):
    #     """Plots the scores."""
    #     plt.figure(figsize=(15, 5))
    #     bins = list(scores.keys())
    #     values = list(scores.values())
    #     bars = plt.bar(bins, values, color='skyblue')
    #     plt.xlabel('Accuracy Bins')
    #     plt.ylabel('Average Success Probability')
    #     plt.suptitle(f'Model Success Probability Across Accuracy Bins')
    #     plt.title(f'Sampled: {self.num_models} models, {self.num_samples} samples')
    #     # Annotate bars with the value of the y-axis
    #     for bar in bars:
    #         yval = bar.get_height()
    #         # plt.text(bar.get_x() + (bar.get_width() / 2), yval, round(yval, 2), va='bottom')  # va: vertical alignment
    #         plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{round(yval, 2)}",  # Adding a small offset (0.01) above the bar for clarity
    #              ha='center', va='bottom')  # 'ha' is horizontal alignment
    #
    #
    #     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    models_default = ["Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat-hf", "OLMo-7B-Instruct-hf"]
    parser.add_argument("--models_to_save", type=list, default=models_default, help="List of models to visualize")
    parser.add_argument("--shot", type=str, default="zero_shot", choices=["zero_shot", "three_shot"], help="Shot type")
    parser.add_argument("--output_path", type=str, default=SamplesAnalysisConstants.SAMPLES_PATH, help="Path to save the samples")
    parser.add_argument("--num_models", type=int, default=1, help="Number of models to sample")
    parser.add_argument("--num_samples", type=int, default=11, help="Number of samples to take from each bin")
    args = parser.parse_args()
    check_prob_samples = checkProbSamples(models_to_save=args.models_to_save, shot=args.shot, output_path=args.output_path,
                                 num_models=args.num_models, num_samples=args.num_samples)
    check_prob_samples.execute()
