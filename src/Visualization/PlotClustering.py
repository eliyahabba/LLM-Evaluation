import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import pyplot as plt


class PlotClustering:
    def __init__(self, data: pd.DataFrame, x: str, y: str, z: str, cluster: str) -> None:
        """
        Initializes the PlotClustering object.
        @param data: the data to plot
        @param x: the x-axis column
        @param y: the y-axis column
        @param z: the z-axis column
        @param cluster: the cluster column
        """
        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.cluster = cluster

    def create_fig(self) -> px.scatter_3d:
        """
        Create the 3D scatter plot figure
        @return: the 3D scatter plot figure
        """
        self.data[self.cluster] = self.data[self.cluster].astype(str)

        fig = px.scatter_3d(
            self.data,
            x=self.x,
            y=self.y,
            z=self.z,
            color=self.cluster,
            # hover_name = "country",
            # log_x = True,
            # size_max = 60,
        )
        return fig

    def plot_cluster(self) -> None:
        """
        Plot the clusters of the data
        @return: None
        """
        fig = self.create_fig()
        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            # Use the Streamlit theme.
            # This is the default. So you can also omit the theme argument.
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with tab2:
            # Use the native Plotly theme.
            st.plotly_chart(fig, theme=None, use_container_width=True)

        fig = self.create_fig()
        st.plotly_chart(fig, theme=None, use_container_width=True)

    def plot_elbow_method(self, data: pd.DataFrame, cluster_columns: list) -> None:
        """
        Plot the elbow method with the distortions
        @param data: The data to plot
        @param cluster_columns: The cluster columns
        @return: None
        """
        data = data[cluster_columns]
        distortions = data.iloc[-1].values
        # plot the elbow method, the distortions
        fig = plt.figure()
        plt.plot(range(len(distortions)), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.xticks(range(len(distortions)), cluster_columns)
        st.pyplot(fig)


if __name__ == "__main__":
    data = pd.read_csv("data/cluster_data.csv")
    plot_clustering = PlotClustering(data, "x", "y", "z", "cluster")
    plot_clustering.plot_cluster()
