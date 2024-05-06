import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants
from src.utils.MMLUConstants import MMLUConstants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
BestCombinationsConstants = Constants.BestCombinationsConstants

BEST_COMBINATIONS_PATH = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH / f"{ResultConstants.BEST_COMBINATIONS}.csv"

MMLU_SUBCATEGORIES = MMLUConstants.SUBCATEGORIES
MMLU_CATEGORIES = MMLUConstants.SUBCATEGORIES_TO_CATEGORIES


class BestCombinationsDisplayer:
    def __init__(self, file_path: Path = BEST_COMBINATIONS_PATH, override_options: dict = None) -> None:
        """
        Initialize the BestCombinationsDisplayer class.

        @param file_path: Path to the CSV file containing the best combinations.
        @param override_options: Dictionary containing options to override histograms.
        """
        self.best_combinations: pd.DataFrame = pd.read_csv(file_path)
        self._filter_best_combinations()
        self.override_options: dict = override_options

    def display_best_combinations(self, model_data: pd.DataFrame) -> None:
        """
        Displays the best combinations for each model and dataset.

        @param model_data: DataFrame containing model and dataset information.
        """
        st.title("Best Combinations")
        st.write("The table below displays the best combinations for each model and dataset on all possible axes.")
        self._add_mmlu_columns(model_data)
        st.write(model_data)

    def _filter_best_combinations(self) -> None:
        """
        Filters the best combinations DataFrame to only include relevant datasets.
        """
        self.best_combinations.drop(columns=[ResultConstants.ACCURACY_COLUMN], inplace=True)
        self.best_combinations = self.best_combinations[
            self.best_combinations[BestCombinationsConstants.DATASET].str.startswith(MMLUConstants.MMLU_CARDS_PREFIX)]

    def _add_mmlu_columns(self, model_data: pd.DataFrame) -> None:
        """
        Adds MMLU subcategories and categories columns to the model data.

        @param model_data: DataFrame containing model data to update.
        """
        model_data[MMLUConstants.SUBCATEGORIES_COLUMN] = model_data[BestCombinationsConstants.DATASET].apply(
            lambda x: (MMLU_SUBCATEGORIES[x.split(MMLUConstants.MMLU_CARDS_PREFIX)[1]][0] if x.startswith(
                MMLUConstants.MMLU_CARDS_PREFIX) else 'None'))
        model_data[MMLUConstants.CATEGORIES_COLUMN] = model_data[BestCombinationsConstants.DATASET].apply(
            lambda x: (
                MMLU_CATEGORIES[MMLU_SUBCATEGORIES[x.split(MMLUConstants.MMLU_CARDS_PREFIX)[1]][0]] if x.startswith(
                    MMLUConstants.MMLU_CARDS_PREFIX) else 'None'))

    def _plot_histograms(self, cur_data: pd.DataFrame, group: str) -> list[plt.Figure]:
        """
        Plots histograms for each axis based on the current data.

        @param cur_data: DataFrame containing the current group data.
        @param group: The group identifier for which to plot histograms.
        @return: List of matplotlib Figures with histograms.
        """
        figs = []
        for axis in self.override_options.keys():
            axis_values = self.override_options[axis]
            for i in range(len(axis_values)):
                if axis_values[i] == '\n':
                    axis_values[i] = '\\n'
                if axis_values[i] == ' ':
                    axis_values[i] = '\\s'

            fig = plt.figure()
            values = cur_data[axis].values

            axis_values_counts = {value: np.sum(values == value) for value in axis_values}
            axis_values_counts = dict(sorted(axis_values_counts.items()))
            unique_values = list(axis_values_counts.keys())
            value_indices = [unique_values.index(value) for value in values]
            if isinstance(unique_values[0], bool):
                unique_values = [str(unique_value) for unique_value in unique_values]

            bins = np.arange(len(unique_values) + 1) - 0.5

            plt.hist(value_indices, bins=bins, alpha=0.75, rwidth=0.8)
            bin_centers = np.arange(len(unique_values))
            plt.xticks(bin_centers, labels=unique_values)
            plt.title(f"{axis} histogram")
            figs.append(fig)
        return figs

    def evaluate_by_model_or_dataset(self, model_or_dataset_key: str, model_data: pd.DataFrame,
                                     split_option: str = None) -> None:
        """
        Evaluates data by the specified key (model or dataset) and splits if required.

        @param model_or_dataset_key: Key to evaluate by (e.g., model).
        @param model_data: DataFrame containing model or dataset data.
        @param split_option: Option to split data into subcategories or categories.
        """
        st.subheader(f"Evaluation by {model_or_dataset_key}")

        if split_option == MMLUConstants.SUBCATEGORIES_COLUMN:
            model_datas = {group: model_data[model_data[MMLUConstants.SUBCATEGORIES_COLUMN] == group] for group
                           in model_data[MMLUConstants.SUBCATEGORIES_COLUMN].unique()}
        elif split_option == MMLUConstants.CATEGORIES_COLUMN:
            model_datas = {group: model_data[model_data[MMLUConstants.CATEGORIES_COLUMN] == group] for group in
                           model_data[MMLUConstants.CATEGORIES_COLUMN].unique()}
        else:
            model_datas = [model_data]

        for group, cur_data in model_datas.items():
            figs = self._plot_histograms(cur_data, group)
            st.write(f"Group: {group}, Number of samples: {len(cur_data)}")
            cols = st.columns(len(figs))
            for i, fig in enumerate(figs):
                with cols[i]:
                    st.pyplot(fig)

    def evaluate(self) -> None:
        """
        Renders the evaluation of the best combinations.

        Displays the best combinations and allows the user to split by model or dataset.
        """
        st.title("Evaluation")
        st.subheader("The table displays the evaluation of the best combinations of each model")
        models = self.best_combinations[BestCombinationsConstants.MODEL].unique()

        model = st.selectbox("Choose a model", models)
        model_data = self.best_combinations[self.best_combinations[BestCombinationsConstants.MODEL] == model]
        model_data = model_data.sort_values(by=BestCombinationsConstants.DATASET).reset_index(drop=True)
        self.display_best_combinations(model_data)
        split_option = st.selectbox("Split the dataset to train and test by:", MMLUConstants.SPLIT_OPTIONS)
        self.evaluate_by_model_or_dataset("model", model_data, split_option)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=BEST_COMBINATIONS_PATH,
                        help="Path to the best combinations file")
    args = parser.parse_args()
    best_combinations_displayer = BestCombinationsDisplayer(args.file_path, ConfigParams.override_options)
    best_combinations_displayer.evaluate()
