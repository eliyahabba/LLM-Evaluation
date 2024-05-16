import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.Visualization.DisplayConfigurationsGroups import DisplayConfigurationsGroups

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants
from src.utils.MMLUConstants import MMLUConstants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
BestCombinationsConstants = Constants.BestCombinationsConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
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
        self.file_path: Path = file_path
        self.results_folder = self.file_path.parent
        self.override_options: dict = override_options

        self.best_combinations: pd.DataFrame = self.read_best_combinations()
        self._filter_best_combinations()
        self._read_templates_metadata()

    def read_best_combinations(self) -> pd.DataFrame:
        """
        Reads the best combinations DataFrame.

        @return: DataFrame containing the best combinations.
        """
        return pd.read_csv(self.file_path)

    def _read_templates_metadata(self):
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / TemplatesGeneratorConstants.TEMPLATES_METADATA
        self.templates_metadata = pd.read_csv(metadata_file, index_col='template_name')

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

    def _create_histogram(self, axis, axis_values, cur_data: pd.DataFrame) -> dict:
        values = cur_data[axis].values
        axis_values_counts = {value: np.sum(values == value) for value in axis_values}
        axis_values_counts = dict(sorted(axis_values_counts.items()))
        return axis_values_counts

    def _create_figure(self, hists: dict) -> list:
        figs = []
        for axis, axis_values_counts in hists.items():
            fig = plt.figure()
            plt.bar(list(axis_values_counts.keys()), list(axis_values_counts.values()))
            plt.title(f"{axis} histogram")
            figs.append(fig)
        return figs

    def _plot_histograms(self, cur_data: pd.DataFrame) -> dict:
        """
        Plots histograms for each axis based on the current data.

        @param cur_data: DataFrame containing the current group data.
        @return: List of matplotlib Figures with histograms.
        """
        hists = {}
        for axis in self.override_options.keys():
            axis_values = self.override_options[axis]
            for i in range(len(axis_values)):
                if axis_values[i] == '\n':
                    axis_values[i] = '\\n'
                if axis_values[i] == ' ':
                    axis_values[i] = '\\s'
            hists[axis] = self._create_histogram(axis, axis_values, cur_data)
        return hists

    def split_data_by_option(self, model_data: pd.DataFrame, split_option: str) -> dict:
        """
        Splits the model data by the specified option.
        @param model_data: 
        @param split_option: 
        @return: 
        """
        if split_option == MMLUConstants.SUBCATEGORIES_COLUMN:
            model_data_splitted = {group: model_data[model_data[MMLUConstants.SUBCATEGORIES_COLUMN] == group] for group
                                   in model_data[MMLUConstants.SUBCATEGORIES_COLUMN].unique()}
        elif split_option == MMLUConstants.CATEGORIES_COLUMN:
            model_data_splitted = {group: model_data[model_data[MMLUConstants.CATEGORIES_COLUMN] == group] for group in
                                   model_data[MMLUConstants.CATEGORIES_COLUMN].unique()}
        else:
            model_data_splitted = [model_data]

        return model_data_splitted

    def choose_group_or_top_k(self) -> Tuple[str, int]:
        group_or_top_k = st.radio("Do you want to take the best k or the first group?", ['Top k', 'First group'])
        top_k = None
        if group_or_top_k == 'Top k':
            top_k = st.number_input("Enter the number of top k groups you want to take", 1, 20, 1)

        return group_or_top_k, top_k

    def evaluate_the_model(self, model: str, model_data: pd.DataFrame,
                           display_histograms: bool = False,
                           split_option: str = None) -> None:
        """
        Evaluates data by the specified key (model or dataset) and splits if required.

        @param model_data: DataFrame containing model or dataset data.
        @param split_option: Option to split data into subcategories or categories.
        """
        model_data_splitted = self.split_data_by_option(model_data, split_option)
        group_or_top_k, top_k = self.choose_group_or_top_k()

        for group_name, cur_data in model_data_splitted.items():
            # read the df of the current group
            dataset_names = cur_data.dataset.values
            datasets_of_the_current_group = self.get_datasets_of_the_current_group(model, group_or_top_k, top_k,
                                                                                   dataset_names)
            # concatenate the dfs to new df
            best_templates_of_cur_group = [df.index.values.tolist() for df in datasets_of_the_current_group]
            # map between the name of template and configuration
            selected_configurations_df = self.convert_templates_names_to_conf_values(best_templates_of_cur_group)
            configurations_counter = self._plot_histograms(selected_configurations_df)
            st.markdown(f'<span style="font-size: 20px; color:blue">{group_name}</span>', unsafe_allow_html=True)

            if display_histograms:
                self.display_bars(configurations_counter, group_name, cur_data)

            most_common_configuration = self.calculate_most_common_configurations(configurations_counter)
            st.markdown(f'<span style="font-size: 17px;">**Best configurations**</span>', unsafe_allow_html=True)
            display_configurations_groups = DisplayConfigurationsGroups(self.results_folder, self.templates_metadata)
            display_configurations_groups.check_the_group_of_conf(most_common_configuration, model,
                                                                  cur_data.dataset.values)
            # add empty line
            st.write("")

    def display_bars(self, hists: dict, group: str, cur_data: pd.DataFrame) -> None:
        figs = self._create_figure(hists)
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
        st.title("Evaluation by model")
        display_histograms = st.checkbox("Display histograms")
        models = self.best_combinations[BestCombinationsConstants.MODEL].unique()

        model = st.selectbox("Choose a model", models)
        model_data = self.best_combinations[self.best_combinations[BestCombinationsConstants.MODEL] == model]
        model_data = model_data.sort_values(by=BestCombinationsConstants.DATASET).reset_index(drop=True)
        self._add_mmlu_columns(model_data)

        split_option = st.selectbox("Split the dataset by:", MMLUConstants.SPLIT_OPTIONS)

        self.evaluate_the_model(model, model_data, display_histograms, split_option)

    def calculate_most_common_configurations(self, hists: dict):
        """
        Calculates the minimum and maximum configurations for each axis based on the histograms.
        @param hists:
        @return:
        """
        max_results = {}
        for axis, data in hists.items():
            max_value = max(data.values())
            # Check for ties in max and min values
            max_keys = [k for k, v in data.items() if v == max_value]
            max_results[axis] = max_keys

        return max_results

    def get_datasets_of_the_current_group(self, model: str, group_or_top_k: str, top_k: int, dataset_names: List[str]):
        dfs = []

        for dataset_name in dataset_names:
            results_folder = self.file_path.parent
            groups_path = results_folder / model / dataset_name / \
                          Path(ResultConstants.ZERO_SHOT) / \
                          Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                          Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
            template_groups_df = pd.read_csv(groups_path)
            if group_or_top_k == 'First group':
                template_groups_df = template_groups_df[template_groups_df['group'] == 'A']
            else:
                template_groups_df = template_groups_df.sort_values(by='accuracy', ascending=False).head(top_k)
            template_groups_df['statistic, pvalue, row is better than column'] = template_groups_df[
                'statistic, pvalue, row is better than column'].map(
                lambda x: x.replace("best set: ", "") if "best set: " in x else x)
            # also replace the columns that contain the string "best set: " in the prefix
            template_groups_df.columns = template_groups_df.columns.map(
                lambda x: x.replace("best set: ", "") if "best set: " in x else x)
            # set index to be ('statistic, pvalue, row is better than column')
            template_groups_df.set_index('statistic, pvalue, row is better than column', inplace=True)
            dfs.append(template_groups_df)
        return dfs

    def convert_templates_names_to_conf_values(self, best_templates_of_cur_group):
        selected_configurations = []
        for df in best_templates_of_cur_group:
            template_cols = [col for col in df if 'template' in col]
            for template_col in template_cols:
                selected_configurations.append(self.templates_metadata.loc[template_col].to_dict())
        selected_configurations_df = pd.DataFrame(selected_configurations)
        return selected_configurations_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=BEST_COMBINATIONS_PATH,
                        help="Path to the best combinations file")
    args = parser.parse_args()
    best_combinations_displayer = BestCombinationsDisplayer(args.file_path, ConfigParams.override_options)
    best_combinations_displayer.evaluate()
