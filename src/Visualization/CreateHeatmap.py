import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

file_path = Path(__file__).parents[2]
sys.path.append(str(file_path))

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from pathlib import Path

import streamlit as st

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class CreateHeatmap:
    def __init__(self, dataset_file_name: str, result_file: Path):
        self.dataset_file_name = dataset_file_name
        self.result_file = result_file

    def create_axis_option(self):
        st.markdown("## Heatmap of the accuracy of the templates")
        override_options = ConfigParams.override_options
        # choose every time 2 params from the override_options to be the axis's
        # in the heatmap, and detemine the value of the other params
        config_options = list(override_options.keys())

        # Display message informing the user to choose exactly 2 options for the axis
        axis_options = st.multiselect("Choose exactly 2 axis options for the heatmap", config_options,
                                      default=config_options[:2], key="axis_selection")

        # Validate that exactly 2 options are selected for the axis
        if len(axis_options) != 2:
            st.error("Please select exactly 2 axis options.")
            # stop the execution of the function
            st.stop()

            return None, None

        # for the others params, add option to choose the value for each one
        # selected_options = [option for option in config_options if option in axis_options]
        selected_values = {}
        for option in config_options:
            if option not in axis_options:
                selected_values[option] = st.selectbox(f"Choose the value for {option}", override_options[option])
        return axis_options, selected_values

    def create_heatmap(self):
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / self.dataset_file_name / "templates_metadata.csv"
        metadata_df = pd.read_csv(metadata_file, index_col='template_name')
        axis_options, selected_values = self.create_axis_option()
        heatmap_df = self.generate_heatmap(metadata_df, axis_options, selected_values)
        # heatmap_df = self.generate_heatmap2(metadata_df, axis_options, selected_values)

        self.plot_heatmap(heatmap_df)

    def plot_heatmap(self, heatmap_df: pd.DataFrame):
        # create visualization for the heatmap with seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        # if the big size of the axis is the rows, we need to rotate the y axis
        if heatmap_df.shape[0] > heatmap_df.shape[1]:
            # rotate the matrix
            heatmap_df = heatmap_df.T

        g = sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt='.2f', annot_kws={"fontsize": 16},
                        linewidths=2, linecolor='black',
                        square=True)
        # add a black line to separate the last row and column
        ax.axvline(x=heatmap_df.shape[1] - 1, color='black', linewidth=6)
        ax.axhline(y=heatmap_df.shape[0] - 1, color='black', linewidth=6)
        g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=14)
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=14)
        # resize the name of the axis
        g.set_ylabel(g.get_ylabel(), fontsize=16, fontweight='bold', color="red")
        g.set_xlabel(g.get_xlabel(), fontsize=16, fontweight='bold', color="red")
        # put the x label on top
        g.xaxis.tick_top()
        g.xaxis.set_label_position('top')
        # add a titlw to the heatmap
        plt.title("Heatmap of the accuracy of the templates", fontsize=17, fontweight='bold')
        # add a gap between the title and the heatmap
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        st.pyplot(fig)

    def generate_heatmap(self, metadata_df: pd.DataFrame, axis_options: list, selected_values: dict) -> pd.DataFrame:
        # choose the relevant rows from the metadata_df
        for option, value in selected_values.items():
            metadata_df = metadata_df[metadata_df[option] == value]

        # now we have the relevant rows, we need to choose the relevant columns for the heatmap
        # create 2d matrix when the rows are the values of the first axis and the columns are the values of the second axis
        # and the values are the accuracy of the template
        df = pd.read_csv(self.result_file)

        # add the 'accuracy' columns from df to the metadata_df by the template_name
        metadata_df = metadata_df.join(df.set_index('template_name')['accuracy'], on='template_name')
        heatmap_df = metadata_df.pivot_table(index=axis_options[0], columns=axis_options[1], values='accuracy')
        # add the average of the accuracy for each row and column
        heatmap_df['Average'] = heatmap_df.mean(axis=1)
        heatmap_df.loc['Average'] = heatmap_df.mean(axis=0)
        # the last row and column cell should be empty
        heatmap_df.loc['Average', 'Average'] = None
        return heatmap_df
