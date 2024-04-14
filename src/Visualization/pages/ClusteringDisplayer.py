from pathlib import Path

import pandas as pd
import streamlit as st

from src.Visualization.PlotClustering import PlotClustering
from src.Visualization.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class ClusteringDisplayer:
    def __init__(self):
        self.dataset = None
        self.dataset_file_name = None

    def display_page(self):
        st.title("Clustering Displayer")
        st.write("Displaying clustering of the accuracy of different templates in the dataset+model")
        self.dataset_file_name, selected_shot_file_name = ResultsLoader.select_experiment_params()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, result_file = ResultsLoader.select_result_file(result_files,
                                                                         ResultConstants.CLUSTERING_RESULTS)

        df = self.read_data(result_file)
        metadata_df = self.read_metadata()
        merged_df = self.merge_data(df, metadata_df)
        self.plot_clusters(merged_df)

    def read_data(self, results_file: Path):
        """
        Display the results of the model.

        @param results_file: the path to the results file
        @return: None
        """
        df = pd.read_csv(results_file, index_col=0)
        df.set_index('template_name', inplace=True)
        return df

    def read_metadata(self):
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / self.dataset_file_name / TemplatesGeneratorConstants.TEMPLATES_METADATA
        metadata_df = pd.read_csv(metadata_file, index_col='template_name')
        return metadata_df

    def plot_clusters(self, data: pd.DataFrame):
        # select the cluster column
        columns = [col for col in data.columns if 'K=' in col]
        k_cluster = st.selectbox("Select the cluster column", columns)
        # select the row that corresponds to the selected cluster
        plot_clustering = PlotClustering(data, "x", "y", "z", k_cluster)
        plot_clustering.plot_cluster()

    def merge_data(self, df, metadata_df):
        # merge the the tables side by side
        merged_df = pd.concat([metadata_df, df], axis=1)
        return merged_df


if __name__ == "__main__":
    ClusteringDisplayer().display_page()
#         # Plotting the histogram
