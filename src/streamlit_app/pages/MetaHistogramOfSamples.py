import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.streamlit_app.ui_components.ResultsLoader import ResultsLoader
from src.streamlit_app.ui_components.SamplesNavigator import SamplesNavigator
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
DatasetsConstants = Constants.DatasetsConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class MetaHistogramOfSamples:
    def __init__(self, dataset_sizes_path):

        self.dataset_sizes_path = dataset_sizes_path

    def display_page(self):
        st.title("Histogram of Samples")
        selected_results_file, model_files, shot = self.get_files()
        self.display_aggregated_results(selected_results_file, model_files, shot)

    def get_model_files(self, selected_results_file):
        folders = [file for file in selected_results_file.iterdir() if file.is_dir()]
        models_names = {f.name: f for f in folders}
        sorted_folders = dict(sorted(models_names.items(), key=lambda x: (x[0].lower(), x[0]), reverse=False))
        models = st.sidebar.multiselect("Select models to visualize", list(sorted_folders.keys()),
                                        default=list(sorted_folders.keys()))
        # add the model to st.session_state
        st.session_state["models"] = models
        selected_models_files = [models_names[model] for model in models]
        return selected_models_files

    def get_files(self):
        main_results_path = ExperimentConstants.MAIN_RESULTS_PATH
        selected_results_file = ResultsLoader.get_folder_selections_options(
            main_results_path, "Select results folder to visualize", reverse=True
        )
        self.results_folder = main_results_path / selected_results_file
        shot = st.sidebar.selectbox("Select number of shots", ["zero_shot", "three_shot"])
        model_files = self.get_model_files(selected_results_file)
        return selected_results_file, model_files, shot

    def display_aggregated_results(self, selected_results_file, selected_models_files, shot):
        datasets_names = self.get_dataset_split_options()
        total_merge_df = MetaHistogramCalculator.aggregate_data_across_models(selected_results_file,
                                                                              selected_models_files, shot,
                                                                              datasets_names)
        total_merge_df = MetaHistogramCalculator.calculate_and_add_accuracy_columns(total_merge_df)
        self.plot_aggregated_histogram(total_merge_df)
        self.display_examples(total_merge_df)

    def get_dataset_split_options(self):
        other_datasets = DatasetsConstants.OTHER
        data_type = st.selectbox("Select datasets", [DatasetsConstants.MMLU_NAME,
                                                     DatasetsConstants.MMLU_PRO_NAME] + other_datasets)
        ds_size = TemplatesGeneratorConstants.DATASET_SIZES_PATH
        df_ds_size = pd.read_csv(ds_size)
        if data_type == DatasetsConstants.MMLU_NAME:
            df_ds_size = df_ds_size[
                ~df_ds_size['Name'].str.startswith(
                    tuple(d for d in [DatasetsConstants.MMLU_NAME, DatasetsConstants.MMLU_PRO_NAME]
                          if d != data_type)
                )]
        df_ds_size = df_ds_size[df_ds_size["Name"].str.startswith(data_type)]
        datasets_names = df_ds_size["Name"].tolist()
        # datasets_names = MMLUSplitter.get_data_files(MMLUConstants.ALL_NAMES,
        #                                              MMLUSplitter.get_data_options(MMLUConstants.ALL_NAMES))
        return datasets_names

    def plot_aggregated_histogram(self, df):
        """
        Plot the histogram of the results.
        @param df: DataFrame containing the data to plot.
        """
        # title = f"Aggregated Histogram by {split_option} {split_option_value}"
        title = f"Aggregated Histogram by"
        st.markdown(
            f"There are {len(df)} examples in the dataset across {int(df['number_of_predictions'].mean())} configurations "
            f"(configuration - combination of model and prompt variation).")
        st.markdown(
            f"The models in the plots are {', '.join(st.session_state['models'])}")
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
        st.pyplot(fig)

    def display_examples(self, df):
        min_percentage = st.slider("Minimum percentage of examples to display", 0, 100, 0, step=5)
        max_percentage = st.slider("Maximum percentage of examples to display", 0, 100, 5, step=5)
        examples = df[(df['accuracy'] > min_percentage) & (df['accuracy'] < max_percentage)]
        self.display_example_details(examples)

    def display_example_details(self, examples):
        # Implementation to display details for each example
        example_data = MetaHistogramCalculator.extract_example_data(examples)
        self.display_bar_chart(example_data)
        self.display_bar_chart_precentages(example_data)  # Precentage Chart
        self.display_text_examples(example_data)

    def display_bar_chart(self, example_data):
        fig = px.bar(example_data['dataset'].value_counts(), x=example_data['dataset'].value_counts().index,
                     y=example_data['dataset'].value_counts().values,
                     labels={'x': 'dataset', 'y': 'number of examples'},
                     title='Number of examples for each dataset in the selected range')
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    def get_number_of_examples_per_topic(self):
        df = pd.read_csv(self.dataset_sizes_path)
        return dict(zip(df['Name'], df['test']))

    def display_bar_chart_precentages(self, example_data):
        try:
            dataset_size = self.get_number_of_examples_per_topic()
            current_count = example_data['dataset'].value_counts()

            percentages = {
                dataset: (current_count[dataset] / dataset_size[dataset] * 100)
                for dataset in current_count.index
                if dataset in dataset_size
            }

            percentage_series = pd.Series(
                percentages).sort_values(ascending=False)

            fig = px.bar(
                x=percentage_series.index,
                y=percentage_series.values,
                labels={'x': 'Dataset', 'y': 'Percentage of Examples'},
                title='Percentage of examples in the selected range from total e',
            )

            fig.update_traces(texttemplate='%{y:.0f}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")

    def display_text_examples(self, example_data):
        dataset = st.selectbox("Select dataset", example_data['dataset'].unique())
        # results_folder = ResultConstants.MAIN_RESULTS_PATH / Path("MultipleChoiceTemplatesStructured")
        default_model = "Mistral-7B-Instruct-v0.2"
        models = sorted(
            [model for model in os.listdir(self.results_folder) if os.path.isdir(self.results_folder / model)])
        model = st.selectbox("Select model", models,
                             index=models.index(default_model) if default_model in models else 0)
        selected_examples = example_data[example_data['dataset'] == dataset]
        # current_instance = selected_examples.iloc[st.session_state.get('file_index', 0)]
        current_instance = self.get_current_index(selected_examples, dataset)
        full_results_path = self.results_folder / model / dataset / "zero_shot" / "empty_system_format" / "experiment_template_0.json"
        with open(full_results_path, "r") as file:
            template = json.load(file)
        sample = template["results"]["test"][current_instance]
        self.display_sample_details(sample)
        st.write("----")

    def get_current_index(self, examples_id, dataset):
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
            [f"sentence {i + 1}" for i in range(0, st.session_state["files_number"])],
            index=st.session_state["file_index"],
            on_change=SamplesNavigator.go_to_sentence,
            key="selected_sentence",
        )

        current_instance = examples_id[examples_id['example_number'] == dataset_index[st.session_state['file_index']]]
        current_instance = int(current_instance.example_number.values[0])
        return current_instance

    def display_sample_details(self, sample):
        formatted_instance = sample['Instance'].replace('\n\n', '<br><br>').replace('\n', '<br>')
        formatted_ground_truth = sample['GroundTruth']
        formatted_prediction = sample['Result']
        formatted_score = sample['Score']

        st.markdown(
            f"**Instance**: {formatted_instance}<br>"
            f"**Ground Truth**: {formatted_ground_truth}<br>"
            f"**Predicted**: {formatted_prediction}<br>"
            f"**Score**: {formatted_score}",
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    dataset_sizes_path = Constants.TemplatesGeneratorConstants.DATASET_SIZES_PATH
    hos = MetaHistogramOfSamples(dataset_sizes_path)
    hos.display_page()
