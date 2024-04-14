from pathlib import Path

import pandas as pd
import streamlit as st

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.Visualization.PlotClustering import PlotClustering
from src.Visualization.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class ClusteringDisplayer:
    def __init__(self):
        self.override_options = ConfigParams.override_options
        self.dataset = None
        self.dataset_file_name = None

    def display_page(self) -> None:
        """
        Display the clustering displayer page
        @return: None
        """
        st.title("Clustering Displayer")
        st.write("Displaying clustering of the accuracy of different templates in the dataset+model")
        self.dataset_file_name, selected_shot_file_name = ResultsLoader.select_experiment_params()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, result_file = ResultsLoader.select_result_file(result_files,
                                                                         ResultConstants.CLUSTERING_RESULTS)

        clustering_data = self.read_clustering_results_data(result_file)
        metadata_df = self.read_metadata()
        merged_df = self.merge_data(clustering_data, metadata_df)
        self.plot_clusters(merged_df)

    def read_clustering_results_data(self, results_file: Path) -> pd.DataFrame:
        """
        Read the clustering results from the specified file
        @param results_file: the path to clustering results file
        @return: a DataFrame containing the clustering results
        """
        df = pd.read_csv(results_file, index_col=0)
        df.set_index('template_name', inplace=True)
        return df

    def read_metadata(self) -> pd.DataFrame:
        """
        Read the metadata of the templates
        @return: a DataFrame containing the metadata of the templates
        """
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / self.dataset_file_name / TemplatesGeneratorConstants.TEMPLATES_METADATA
        metadata_df = pd.read_csv(metadata_file, index_col='template_name')
        return metadata_df

    def plot_clusters(self, data: pd.DataFrame) -> None:
        """
        Plot the clusters of the data
        @param data: the data to plot
        @return: None
        """
        # select the cluster column
        cluster_columns = [col for col in data.columns if 'K=' in col]
        k_cluster = st.selectbox("Select the cluster column", cluster_columns)
        # select the row that corresponds to the selected cluster
        # take the other columns as the axis columns (all columns except the cluster_columns)
        axis_columns = [col for col in data.columns if col not in cluster_columns]

        select_axis_title = "Select 3 columns to plot the clusters"
        selected_axis = st.multiselect(select_axis_title,
                                       list(self.override_options.keys()),
                                       default=list(self.override_options.keys())[:3],
                                       max_selections=3,
                                       key="selected_columns")
        x, y, z = selected_axis
        plot_clustering = PlotClustering(data, x=x, y=y, z=z, cluster=k_cluster)
        plot_clustering.plot_cluster()

    def merge_data(self, clustering_data: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the clustering data with the metadata
        @param clustering_data: The clustering data
        @param metadata_df: The metadata of the templates
        @return: a DataFrame containing the merged data
        """
        merged_df = pd.concat([metadata_df, clustering_data], axis=1)
        return merged_df


if __name__ == "__main__":
    ClusteringDisplayer().display_page()
#         # Plotting the histogram
