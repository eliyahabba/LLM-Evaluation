import argparse
import json

import pandas as pd

from src.experiments.data_loading.MMLUSplitter import MMLUSplitter
from src.streamlit_app.ui_components.MetaHistogramCalculator import MetaHistogramCalculator
from src.streamlit_app.ui_components.SamplesNavigator import SamplesNavigator
from src.utils.Constants import Constants

MMLUConstants = Constants.MMLUConstants
ResultConstants = Constants.ResultConstants
SamplesAnalysisConstants = Constants.SamplesAnalysisConstants
ExperimentConstants = Constants.ExperimentConstants
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
import pandas as pd
import streamlit as st


class SampleViewer:
    def __init__(self, sample_path):
        self.sample_df = None
        self.sample_path = sample_path

    def run(self):
        st.title('Sample Viewer')
        self.sample_df = pd.read_csv(self.sample_path)
        datasets = self.load_datasets()
        selected_dataset = st.selectbox("Select a Dataset", datasets)

        if selected_dataset:
            examples = self.select_dataset_examples(selected_dataset)
            selected_index = self.get_current_index(examples, selected_dataset)

            if selected_index or selected_index == 0:
                self.display_results(selected_dataset, selected_index)
    def load_datasets(self):
        # Retrieve unique datasets from the dataframe
        return self.sample_df['Dataset'].unique()

    def select_dataset_examples(self, dataset):
        # Filter dataframe by dataset and return unique indices
        return self.sample_df[self.sample_df['Dataset'] == dataset]['Index'].unique()

    def display_results(self, dataset, index):
        # Filter data for the selected dataset and index
        cur_dataset = self.sample_df[(self.sample_df['Dataset'] == dataset)]
        # map between cur_dataset["Index"] (take unique values) and create new index to be used in the display
        cur_dataset["Index2"] = cur_dataset["Index"].map({k: i for i, k in enumerate(cur_dataset["Index"].unique())})
        data = cur_dataset[cur_dataset['Index2'] == index]

        # Display the Ground Truth, which is common for all entries
        # st.markdown(f"<div style='font-size: 24px;'>{ground_truth}</div>", unsafe_allow_html=True)

        # Display the Instance, which is the same for all models
        instance = data.iloc[0]['Instance']
        st.write(f"### Instance:")
        st.markdown(f"{instance}")

        # Display each model's result in a column
        col_count = len(data['model'].unique())
        cols = st.columns(col_count)

        colors = ["#FF5733", "#33C1FF", "#D633FF", "#FFC733"]  # Define a list of colors
        for i, (idx, row) in enumerate(data.iterrows()):
            with cols[i]:
                st.markdown(f"###### {row['model']}")
                result_formatted = row['Result'].strip().replace('\n', '<br>') # Replace newlines with HTML breaks
                st.markdown(f"<div style='color: {colors[i % len(colors)]};'>{result_formatted}</div>", unsafe_allow_html=True)
                st.write(f"Score: {row['Score']}")

        ground_truth = data.iloc[0]['GroundTruth']
        st.write("### Ground Truth")
        st.markdown(f"```{ground_truth}```")  # Use markdown code block for large text

    def get_current_index(self, examples_id, dataset):
        # dataset_index = examples_id[examples_id['Dataset'] == dataset]['example_number'].values

        # sort the files by the number of the experiment
        # write on the center of the page
        st.markdown(f"#### {len(examples_id)} Examples", unsafe_allow_html=True)
        if "file_index" not in st.session_state:
            st.session_state["file_index"] = 0
        if "dataset" not in st.session_state:
            st.session_state["dataset"] = dataset
        if dataset != st.session_state["dataset"]:
            st.session_state["file_index"] = 0
            st.session_state["dataset"] = dataset

        st.session_state["files_number"] = len(examples_id)

        # add bottoms to choose example
        col1, col2 = st.columns(2)
        with col1:
            st.button(label="Previous sentence", on_click=SamplesNavigator.previous_sentence)
        with col2:
            st.button(label="Next sentence", on_click=SamplesNavigator.next_sentence)
        st.selectbox(
            "Sentences",
            [f"sentence {i+1}" for i in range(0, st.session_state["files_number"])],
            index=st.session_state["file_index"],
            on_change=SamplesNavigator.go_to_sentence,
            key="selected_sentence",
        )

        current_instance = st.session_state['file_index']
        # current_instance = int(current_instance.example_number.values[0])
        return current_instance

if __name__ == "__main__":
    viewer = SampleViewer(SamplesAnalysisConstants.SAMPLES_PATH)
    viewer.run()
