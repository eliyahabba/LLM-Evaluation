import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants
SamplesAnalysisConstants = Constants.SamplesAnalysisConstants
ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


def format_integer(value):
    """Format the value as an integer if not NaN, otherwise as an empty string."""
    if pd.isna(value):
        return ""
    else:
        return f"{int(value)}"


class checkProbSamples:
    def __init__(self, shot):
        self.shot = shot
        self.num_models = st.number_input("Enter the number of models", 1, 10, 3)
        self.num_samples_for_each_bin = st.number_input("Enter the number of samples for each bin", 1, 10, 4)
        self.num_samples = self.num_samples_for_each_bin
        scores_columns = [f"{i * 5}-{(i - 1) * 5}" for i in range(20, 0, -1)]
        scores_columns = [f"{i * 5}-{(i + 1) * 5}" for i in range(0, 20, 1)]
        # ask the user to choose start bin from scores_columns and end bin  from scores_columns
        self.start_bin_label = st.selectbox("Select the start bin", scores_columns[::-1], index=0)
        self.end_bin_label = st.selectbox("Select the end bin", scores_columns[::-1], index=5)
        self.start_bin = scores_columns.index(self.start_bin_label) + 1
        self.end_bin = scores_columns.index(self.end_bin_label) + 1

    def execute(self):
        results_folder, models_files = self.retrieve_files()
        total_aggregated_df = self.aggregate_results(results_folder, models_files)
        self.add_bins(total_aggregated_df)
        # Adjust to multiple start bins
        all_scores = []
        for start_bin in tqdm(range(self.start_bin, self.end_bin, -1)):  # From 20 to 10
            scores = self.evaluate_models_across_bins(total_aggregated_df, start_bin=start_bin, end_bin=0,
                                                      num_models=self.num_models, num_samples=self.num_samples)
            # Normalize length of scores to max length (20 elements)
            for i in range(len(scores)):
                scores[i] = [scores[i][0]] + [np.nan] * (21 - len(scores[i])) + scores[i][1:]
            all_scores.extend(scores)

        scores_df = pd.DataFrame(all_scores)
        sample_plus_model_col = "Sample + Model"
        scores_columns = [f"{i * 5}-{(i - 1) * 5}" for i in range(20, 0, -1)]
        scores_df.columns = [sample_plus_model_col] + scores_columns
        scores_df.set_index(sample_plus_model_col, inplace=True)
        # convert all the non nan values to int
        # self.plot_heatmap_px(scores_df)
        self.plot_heatmap(scores_df)

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
        total_aggregated_df['accuracy_bin'] = pd.cut(total_aggregated_df['accuracy'], bins=bins, labels=labels,
                                                     right=False)

    def evaluate_models_across_bins(self, df, start_bin, end_bin, num_models, num_samples=1000):
        """Evaluates model performance across bins."""
        scores = []

        current_label = f"{(start_bin - 1) * 5}-{start_bin * 5}"
        current_bin_rows = df[df['accuracy_bin'] == current_label]
        num_samples = min(num_samples, len(current_bin_rows))
        sample_current_bin_rows = current_bin_rows.sample(n=num_samples)

        for index, row in sample_current_bin_rows.iterrows():
            # Filter for models that scored 1 in the current example
            successful_models = [col for col in df.columns if 'template' in col and row[col] == 1]

            # Determine the number of models to sample
            sample_size = min(num_models, len(successful_models))

            if sample_size == 0:
                continue  # If no successful models, continue to next row

            # Sample models from the successful ones
            sampled_models = np.random.choice(successful_models, sample_size, replace=False)
            for model in sampled_models:
                row_score = [1]
                sample_plus_model = f"{index}_{model}"
                for current_bin in range(start_bin - 1, end_bin, -1):
                    # Convert bins to labels
                    previous_label = f"{(current_bin - 1) * 5}-{current_bin * 5}"
                    previous_bin_rows = df[df['accuracy_bin'] == previous_label]
                    # Count successful models
                    success_count = 0
                    if row[model] == 1 and not previous_bin_rows[model].empty and previous_bin_rows[model].iloc[0] == 1:
                        success_count = 1
                    row_score.append(success_count)
                # add sample_plus_model to first position of the row_score and move the rest to the right
                row_score = [sample_plus_model] + row_score
                scores.append(row_score)

        return scores

    def plot_heatmap_px(self, score_matrix: pd.DataFrame):
        """Plot a heatmap of model success rates using Plotly for better interactivity and readability."""
        import plotly.graph_objects as go
        x_labels_reversed = score_matrix.columns  # Reverses the label order
        x_positions = list(range(len(x_labels_reversed)))[::-1]

        fig = go.Figure(data=go.Heatmap(
            x=x_positions,
            y=score_matrix.index,
            z=score_matrix.values,
            colorscale='Viridis'))
        # fig.update_xaxes(autorange="reversed")  # Automatically reverse the x-axis range
        fig.update_xaxes(
            autorange="reversed",
            tickvals=x_positions,  # Set tick positions
            ticktext=[f"{label}" for label in x_labels_reversed]  # Set custom tick labels
        )
        fig.update_yaxes(
            autorange="reversed",
            # tickvals=x_positions,  # Set tick positions
            # ticktext=[f"Label {label}" for label in x_labels_reversed]  # Set custom tick labels
        )

        fig.update_layout(
            title='Model Success Rates Across Bins',
            xaxis_title='Accuracy Bins (20 to 1)',
            yaxis_title='Sample Index',
            width=900,  # You can adjust the width and height to suit your display
            height=600,
            paper_bgcolor='white',  # Background color for the outer area
            plot_bgcolor='white',  # Background color for the plot area
            margin=dict(l=70, r=70, t=70, b=70)  # Adjust margins to fit labels
        )

        st.plotly_chart(fig)

    def plot_heatmap(self, score_matrix: pd.DataFrame):
        st.write('Plotting heatmap')
        """Plot a heatmap of model success rates."""
        # Set up the figure size for a better display in Streamlit
        plt.figure(figsize=(20, 15))  # Adjusted to a more reasonable size for web display
        # Create the heatmap with seaborn
        data_for_plot = np.nan_to_num(score_matrix.values, nan=-1)  # Replace NaN with -1
        mask = np.isclose(data_for_plot, -1)  # Create a mask where the data was NaN

        plt.figure(figsize=(20, 15))  # Adjust size as needed

        # Create the heatmap with seaborn, using the adjusted data and mask
        ax = sns.heatmap(data_for_plot, annot=True, cmap='coolwarm', fmt=".0f", mask=mask,
                         xticklabels=score_matrix.columns,  # Set x labels to DataFrame columns
                         yticklabels=score_matrix.index)
        # ax = sns.heatmap(score_matrix, annot=True, cmap='coolwarm', fmt="d", mask=score_matrix.isna())
        # Set titles and labels
        plt.title('Model Success Rates Across Bins')
        plt.xlabel('Accuracy Bins')
        plt.ylabel('Sample Index')
        st.pyplot(plt)

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
    parser.add_argument("--num_models", type=int, default=3, help="Number of models to sample")
    parser.add_argument("--shot", type=str, default="zero_shot", choices=["zero_shot", "three_shot"], help="Shot type")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to take from each bin")
    args = parser.parse_args()
    check_prob_samples = checkProbSamples(shot=args.shot)
    check_prob_samples.execute()
