import pandas as pd
import plotly.express as px
import streamlit as st


class PlotClustering:
    def __init__(self, data: pd.DataFrame, x: str, y: str, z: str, cluster: str):
        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.cluster = cluster

    def create_fig(self):
        self.data[self.cluster] = self.data[self.cluster].astype(str)

        fig = px.scatter_3d(
            self.data,
            x="enumerator",
            y="choices_separator",
            z="shuffle_choices",
            color=self.cluster,
            # hover_name = "country",
            # log_x = True,
            # size_max = 60,
        )
        return fig

    def plot_cluster(self):
        fig = self.create_fig()
        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            # Use the Streamlit theme.
            # This is the default. So you can also omit the theme argument.
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with tab2:
            # Use the native Plotly theme.
            st.plotly_chart(fig, theme=None, use_container_width=True)


if __name__ == "__main__":
    data = pd.read_csv("data/cluster_data.csv")
    plot_clustering = PlotClustering(data, "x", "y", "z", "cluster")
    plot_clustering.plot_cluster()
