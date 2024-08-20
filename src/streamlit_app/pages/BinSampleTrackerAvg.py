import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import plotly.express as px

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.utils.Constants import Constants
import streamlit as st
MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants
SamplesAnalysisConstants = Constants.SamplesAnalysisConstants
ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
LLMProcessorConstants = Constants.LLMProcessorConstants

def format_integer(value):
    """Format the value as an integer if not NaN, otherwise as an empty string."""
    if pd.isna(value):
        return ""
    else:
        return f"{int(value)}"


class BinSampleTrackerAvg:
    def __init__(self, models_to_save, shot, output_path, num_models, num_samples):
        self.models_to_save = models_to_save
        self.shot = shot
        self.output_path = output_path
        self.num_models = num_models
        self.num_models = st.number_input("Enter the number of configurations", 1, 1000, 3)
        self.num_samples = num_samples

    def execute(self):
        results_folder, models_files = self.retrieve_files()
        total_aggregated_df = self.aggregate_results(results_folder, models_files)
        self.add_bins(total_aggregated_df)
        # Adjust to multiple start bins
        all_scores = []
        # self.select_only_models
        self.remove_nan = st.checkbox("Remove models with NaN values", value=True)

        scores_df = self.evaluate_models_across_bins(total_aggregated_df, start_bin=20, end_bin=0,
                                                   num_models=self.num_models)
        self.plot_heatmap(scores_df)
        # self.plot_heatmap2(scores_df)
        # Normalize length of scores to max length (20 elements)
        # sample_plus_model_col = "Sample + Model"
        # scores_columns = [f"{i * 5}-{(i - 1) * 5}" for i in range(20, 0, -1)]
        # scores_df.columns = [sample_plus_model_col] + scores_columns
        # scores_df.set_index(sample_plus_model_col, inplace=True)
        # convert all the non nan values to int

    def plot_heatmap2(self, score_matrix: pd.DataFrame):
        st.write('Plotting heatmap')
        import plotly.graph_objects as go
        # Replace NaN with a specific value and set up the mask
        data_for_plot = np.nan_to_num(score_matrix.values, nan=-1)  # Replace NaN with -1
        mask = np.isclose(data_for_plot, -1)  # Create a mask where the data was NaN

        # Create the heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=data_for_plot,
            x=score_matrix.columns,
            y=score_matrix.index,
            showscale=True,  # To show the color scale
            zmin=-1,  # Set the scale minimum to -1 to manage the masking
            zmax=score_matrix.values.max(),  # Set the scale maximum based on your data
            colorscale='RdBu',  # Color scale
            hoverongaps=False  # Shows hover information even for masked values
        ))

        # Update the layout to add scrolling
        fig.update_layout(
            title='Model Success Rates Across Bins',
            xaxis_title='Accuracy Bins',
            yaxis_title='Sample Index',
            autosize=False,
            height=600,  # Set a fixed height for the chart area
            width=900,  # Set a fixed width for the chart area
            margin=dict(t=50, l=50, r=50, b=50),  # Adjust margins to fit labels
            yaxis=dict(  # Adjust y-axis configuration for vertical scrolling
                autorange=True,
                showgrid=True,  # To show grid lines
                dtick=1,  # Distance between ticks
                tickmode='array',  # Allows for an array of tickvals if needed
                tickvals=score_matrix.index  # Use index for tick values if your index is non-numeric
            )
        )

        # Display the Plotly heatmap in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def plot_heatmap(self, score_matrix: pd.DataFrame):
        st.write('Plotting heatmap')
        # """Plot a heatmap of model success rates."""
        # Set up the figure size for a better display in Streamlit
        plt.figure(figsize=(60, 60))  # Adjusted to a more reasonable size for web display
        # Create the heatmap with seaborn
        data_for_plot = np.nan_to_num(score_matrix.values, nan=-1)  # Replace NaN with -1
        mask = np.isclose(data_for_plot, -1)  # Create a mask where the data was NaN
        # sns.set(font_scale=1.5)  # Adjust this to scale font sizes of labels and annotations

        # Create the heatmap with seaborn, using the adjusted data and mask
        ax = sns.heatmap(data_for_plot, annot=True, cmap='coolwarm', fmt=".1f", mask=mask,
                         xticklabels=score_matrix.columns,  # Set x labels to DataFrame columns
                         yticklabels=score_matrix.index,
                         annot_kws={"fontsize": 24}
                         )
        # ax = sns.heatmap(score_matrix, annot=True, cmap='coolwarm', fmt="d", mask=score_matrix.isna())
        # Set titles and labels
        # plt.title('Model Success Rates Across Bins', fontsize=24)  # Set the size of the title
        # plt.xlabel('Accuracy Bins', fontsize=20)  # Set the size of the x-axis label
        # plt.ylabel('Sample Index', fontsize=20)  # Set the size of the y-axis label

        # Adjust the tick label size
        ax.tick_params(axis='x', labelsize=33)  # Set the fontsize of the x-axis tick labels
        ax.tick_params(axis='y', labelsize=33)  # Set the fontsize of the y-axis tick labels

        plt.title('Model Success Rates Across Bins')
        plt.xlabel('Accuracy Bins')
        plt.ylabel('Sample Index')
        st.pyplot(plt)

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
        # sort by index
        total_aggregated_df = total_aggregated_df.sort_index()
        return total_aggregated_df

    def add_bins(self, total_aggregated_df):
        bins = list(range(0, 101, 5))
        labels = [f"{i}-{i + 5}" for i in range(0, 100, 5)]
        total_aggregated_df['accuracy_bin'] = pd.cut(total_aggregated_df['accuracy'], bins=bins, labels=labels,
                                                     right=False)

    # def evaluate_models_across_bins(self, df, start_bin, end_bin, num_models, num_samples=1000):
    #     """Evaluates model performance across bins."""
    #     scores = []
    #
    #     current_label = f"{(start_bin - 1) * 5}-{start_bin * 5}"
    #     current_bin_rows = df[df['accuracy_bin'] == current_label]
    #     num_samples = min(num_samples, len(current_bin_rows))
    #     sample_current_bin_rows = current_bin_rows.sample(n=num_samples)
    #
    #     for index, row in sample_current_bin_rows.iterrows():
    #         # Filter for models that scored 1 in the current example
    #         successful_models = [col for col in df.columns if 'template' in col and row[col] == 1]
    #
    #         # Determine the number of models to sample
    #         sample_size = min(num_models, len(successful_models))
    #
    #         if sample_size == 0:
    #             continue  # If no successful models, continue to next row
    #
    #         # Sample models from the successful ones
    #         sampled_models = np.random.choice(successful_models, sample_size, replace=False)
    #         for model in sampled_models:
    #             row_score = [1]
    #             sample_plus_model = f"{index}_{model}"
    #             for current_bin in range(start_bin - 1, end_bin, -1):
    #                 # Convert bins to labels
    #                 previous_label = f"{(current_bin - 1) * 5}-{current_bin * 5}"
    #                 previous_bin_rows = df[df['accuracy_bin'] == previous_label]
    #                 # Count successful models
    #                 success_count = 0
    #                 if row[model] == 1 and not previous_bin_rows[model].empty and previous_bin_rows[model].iloc[0] == 1:
    #                     success_count = 1
    #                 row_score.append(success_count)
    #             # add sample_plus_model to first position of the row_score and move the rest to the right
    #             row_score = [sample_plus_model] + row_score
    #             scores.append(row_score)
    #
    #     return scores

    def evaluate_models_across_bins(self, df, start_bin, end_bin, num_models):
        """Evaluates model performance across bins for a sampled subset of models."""
        model_columns = [col for col in df.columns if 'template' in col]

        model_columns = [col for col in df.columns if 'template' in col]
        model_columns_names = list(set([col.split('_experiment')[0] for col in model_columns]))
        model_names = [model.split("/")[-1] for model in list(LLMProcessorConstants.MODEL_NAMES.values())]
        model_names = [model for model in model_names if model in model_columns_names]

        # Streamlit widget to allow users to select models from a list
        st.subheader("Select Models for Analysis")
        selected_models = st.multiselect("Choose models to include in the sampling:", options=model_names,
                                         default=model_names)

        # Filter model_columns to include only those selected by the user
        # This assumes each selected model is a prefix in the respective column names
        filtered_model_columns = [col for col in model_columns if any(model in col for model in selected_models)]

        #add exapnder of slecet specific models:
        #model_names = [model.split("/")[-1] for model in  list(LLMProcessorConstants.MODEL_NAMES.values())]
        #model_columns_names = list(set([col.split('_experiment')[0] for col in model_columns]))
        #model_names = [model for model in model_names if model in model_columns_names]
        # ask the user to select the models(in exapnder) to save and the before sampling filter the columns that are not in the selected models (need the prefix of the columns)
        # sampled_models = np.random.choice(model_columns, min(num_models, len(model_columns)), replace=False)
        sampled_models = np.random.choice(filtered_model_columns, min(num_models, len(filtered_model_columns)),
                                          replace=False)

        # Initialize a dictionary to hold accuracy data for each model across bins
        accuracy_data = {model: [] for model in sampled_models}
        models_with_nan = []
        # Iterate over each bin
        for current_bin in range(start_bin, end_bin, -1):
            current_label = f"{(current_bin - 1) * 5}-{current_bin * 5}"
            current_bin_rows = df[df['accuracy_bin'] == current_label]

            # Calculate the accuracy for each model in the current bin
            for model in sampled_models:
                model_data = current_bin_rows[model].dropna()  # Drop NaN values from the model's data
                if len(current_bin_rows) > 0:
                    # Count successes and calculate accuracy
                    success_count = model_data.sum()
                    total_possible = len(model_data)
                    accuracy = (success_count / total_possible) * 100
                    accuracy = round(accuracy, 2)
                    #
                else:
                    accuracy = 0  # No data for this bin
                    models_with_nan.append(model)
                accuracy_data[model].append(accuracy)

        # Convert the dictionary to a DataFrame for easier viewing/manipulation
        # accuracy_df = accuracy_df.transpose()  # Transpose to have bins as rows and models as columns
        # Add baseline model data
        # baseline_data = [(b - 1) * 5 + 2.5 for b in range(start_bin, end_bin, -1)]
        # accuracy_data['Baseline Model'] = baseline_data
        # put the base line model in the first position
        # accuracy_data = {key: accuracy_data[key] for key in ['Baseline Model'] + list(accuracy_data.keys())}

        accuracy_df = pd.DataFrame.from_dict(accuracy_data, orient='index', columns=[f"{(b - 1) * 5}-{b * 5}" for b in
                                                                                     range(start_bin, end_bin , -1)])

        # remove the models with nan values base on the  models_with_nan list
        # ask the user if remove the models with nan values
        if self.remove_nan:
            accuracy_df = accuracy_df.drop(models_with_nan, axis=0)
        # sort the rows  by the values of the first column
        accuracy_df = accuracy_df.sort_values(by=[f"{(start_bin - 1) * 5}-{start_bin * 5}"], ascending=False)
        # put the base line model (row) in the first position
        # add new row in the first position
        baseline_data = [(b - 1) * 5 + 2.5 for b in range(start_bin, end_bin, -1)]
        accuracy_df.loc['Baseline Model'] = baseline_data

        accuracy_df = accuracy_df.reindex(['Baseline Model'] + list(accuracy_df.index))

        return accuracy_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    models_default = ["Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat-hf", "OLMo-7B-Instruct-hf"]
    parser.add_argument("--models_to_save", type=list, default=models_default, help="List of models to visualize")
    parser.add_argument("--shot", type=str, default="zero_shot", choices=["zero_shot", "three_shot"], help="Shot type")
    parser.add_argument("--output_path", type=str, default=SamplesAnalysisConstants.SAMPLES_PATH,
                        help="Path to save the samples")
    parser.add_argument("--num_models", type=int, default=3, help="Number of models to sample")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to take from each bin")
    args = parser.parse_args()
    check_prob_samples = BinSampleTrackerAvg(models_to_save=args.models_to_save, shot=args.shot,
                                          output_path=args.output_path,
                                          num_models=args.num_models, num_samples=args.num_samples)
    check_prob_samples.execute()
