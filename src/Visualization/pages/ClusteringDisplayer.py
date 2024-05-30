import sys
from pathlib import Path

import pandas as pd
import streamlit as st

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.Visualization.PlotClustering import PlotClustering
from src.Visualization.ResultsLoader import ResultsLoader
from src.utils.Constants import Constants

ResultConstants = Constants.ResultConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ClusteringConstants = Constants.ClusteringConstants


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
        self.dataset_file_name, selected_shot_file_name, _ = ResultsLoader.select_expeCriment_params()

        # find the csv file in the folder if exists
        result_files = ResultsLoader.get_result_files(selected_shot_file_name)
        result_file_name, performance_summary_path = ResultsLoader.select_result_file(result_files,
                                                                                      ResultConstants.
                                                                                      PERFORMANCE_SUMMARY)
        performance_summary_df  = pd.read_csv(performance_summary_path)
        cols_to_remove = ['card', 'system_format', 'score', 'score_name']
        if 'groups_mean_score' in df.columns:
            cols_to_remove.append('groups_mean_score')
        columns_order = ['template_name', 'number_of_instances', 'accuracy', 'accuracy_ci_low', 'accuracy_ci_high',
                         'score_ci_low', 'score_ci_high', 'groups_mean_score']
        performance_summary_df = performance_summary_df.drop(cols_to_remove, axis=1)
        performance_summary_df = performance_summary_df[columns_order]

        clustering_files = [f for f in result_files if ResultConstants.CLUSTERING_RESULTS in f.name]
        clustering_methods = ClusteringConstants.CLUSTERING_METHODS
        # map between the clustering results file name and the CLUSTERING_METHODS name from Constants, so we can
        # display the clustering method name in the select box and get the file name from the map
        clustering_files_map = {i: [f for f in clustering_files if i in f.name.lower()][0] for i in
                                clustering_methods if [f for f in clustering_files if i in f.name.lower()]}

        clustering_method_name = st.selectbox("Select the clustering method to visualize",
                                              key="clustering_method",
                                              options=list(clustering_files_map.keys()))
        clustering_file = clustering_files_map[clustering_method_name]
        clustering_data = self.read_clustering_results_data(clustering_file)
        metadata_df = self.read_metadata()
        merged_df = self.merge_data(clustering_data, metadata_df)
        # add to performance_summary_df the k columns form merged_df when usinf the index
        performance_summary_df.set_index('template_name', inplace=True)
        merged_df_k_columns = merged_df.filter(regex='K=\d+')

        performance_summary_df2 = pd.concat([merged_df_k_columns, performance_summary_df], axis=1)
        self.plot_clusters(merged_df, performance_summary_df2)

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

    def plot_clusters(self, data: pd.DataFrame, performance_summary_df2) -> None:
        """
        Plot the clusters of the data
        @param data: The data to plot
        @return: None
        """
        # select the cluster column
        cluster_columns = [col for col in data.columns if (col not in ConfigParams.override_options.keys())]
        # convert all the cluster columns to int (all the Not Nan values are integers)
        data[cluster_columns] = data[cluster_columns].astype(int)
        k_cluster = st.selectbox("Select the cluster column", cluster_columns)
        # select the row that corresponds to the selected cluster
        # take the other columns as the axis columns (all columns except the cluster_columns)
        axis_columns = [col for col in data.columns if col not in cluster_columns]

        select_axis_title = "Select 3 columns to plot the clusters"
        selected_axis = st.multiselect(select_axis_title,
                                       axis_columns,
                                       default=list(self.override_options.keys())[:3],
                                       max_selections=3,
                                       key="selected_columns")
        x, y, z = selected_axis
        # delete data with negative k_cluster value
        data = data[data[k_cluster] >= 0]
        # take the only index that start with "template" and not others like "generated"
        template_data = data[data.index.str.startswith("template")]
        # sort the df by the cluster column values
        # template_data = template_data.sort_values(by=k_cluster)
        plot_clustering = PlotClustering(template_data, x=x, y=y, z=z, cluster=k_cluster)
        plot_clustering.plot_cluster()
        plot_clustering.display_results_table(performance_summary_df2, k_cluster)

        # check if the last index starts with "Distortions" to plot the elbow method (Kmeans)
        if data.index[-1].startswith("Distortions"):
            # plot the elbow method with the distortions
            plot_clustering.plot_elbow_method(data, cluster_columns)

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
