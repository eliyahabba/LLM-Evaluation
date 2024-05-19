import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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

    def _plot_histograms(self, cur_data: pd.DataFrame, group: str) -> dict:
        """
        Plots histograms for each axis based on the current data.

        @param cur_data: DataFrame containing the current group data.
        @param group: The group identifier for which to plot histograms.
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

        if split_option == MMLUConstants.SUBCATEGORIES_COLUMN:
            model_datas = {group: model_data[model_data[MMLUConstants.SUBCATEGORIES_COLUMN] == group] for group
                           in model_data[MMLUConstants.SUBCATEGORIES_COLUMN].unique()}
        elif split_option == MMLUConstants.CATEGORIES_COLUMN:
            model_datas = {group: model_data[model_data[MMLUConstants.CATEGORIES_COLUMN] == group] for group in
                           model_data[MMLUConstants.CATEGORIES_COLUMN].unique()}
        else:
            model_datas = [model_data]

        return model_datas

    def evaluate_by_model_or_dataset(self, model_or_dataset_key: str, model: str, model_data: pd.DataFrame,
                                     display_histograms: bool = False,
                                     split_option: str = None) -> None:
        """
        Evaluates data by the specified key (model or dataset) and splits if required.

        @param model_or_dataset_key: Key to evaluate by (e.g., model).
        @param model_data: DataFrame containing model or dataset data.
        @param split_option: Option to split data into subcategories or categories.
        """
        model_datas = self.split_data_by_option(model_data, split_option)
        st.subheader(f"Evaluation by {model_or_dataset_key}")
        group_or_top_k = st.radio("Do you want to take the best k or the first group?", ['Top k', 'First group'])
        top_k = None
        if group_or_top_k == 'Top k':
            top_k = st.number_input("Enter the number of top k groups you want to take", 1, 20, 1)
        for group, cur_data in model_datas.items():
            # read the df of the current group
            dataset_names = cur_data.dataset.values
            dfs_of_the_cur_group = self.get_dfs_of_the_cur_group(model, group_or_top_k, top_k, dataset_names)
            # concatenate the dfs to new df
            dfs_cur_group = [df.index.values.tolist() for df in dfs_of_the_cur_group]
            # map between the name of template and configuration
            template_to_conf = self.get_template_to_conf(dfs_cur_group)
            hists = self._plot_histograms(template_to_conf, group)
            st.markdown(f'<span style="font-size: 20px; color:blue">{group}</span>', unsafe_allow_html=True)
            if display_histograms:
                self.display_bar(hists, group, cur_data)

            _, max_group = self.calculate_min_max_configurations_with_more(hists)
            # wrtie with the name of the group blue color

            # check if there is None in min_group.values()
            if None not in max_group.values():
                st.markdown(f'<span style="font-size: 17px;">**Worst configurations**</span>', unsafe_allow_html=True)
                self.check_group_of_conf(max_group, model, cur_data.dataset.values)
            # add empty line
            st.write("")

    def display_bar(self, hists: dict, group: str, cur_data: pd.DataFrame) -> None:
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
        st.title("Evaluation")
        display_histograms = st.checkbox("Display histograms")
        st.subheader("The table displays the evaluation of the best combinations of each model")
        models = self.best_combinations[BestCombinationsConstants.MODEL].unique()

        model = st.selectbox("Choose a model", models)
        model_data = self.best_combinations[self.best_combinations[BestCombinationsConstants.MODEL] == model]
        model_data = model_data.sort_values(by=BestCombinationsConstants.DATASET).reset_index(drop=True)
        self._add_mmlu_columns(model_data)
        split_option = st.selectbox("Split the dataset by:", MMLUConstants.SPLIT_OPTIONS)
        self.evaluate_by_model_or_dataset("model", model, model_data, display_histograms, split_option)

    def calculate_min_max_configurations_with_more(self, hists: dict):
        """
        Calculates the minimum and maximum configurations for each axis based on the histograms.
        @param hists:
        @return:
        """
        min_results = {}
        max_results = {}
        for axis, data in hists.items():
            max_value = max(data.values())
            min_value = min(data.values())

            # Check for ties in max and min values
            max_keys = [k for k, v in data.items() if v == max_value]
            min_keys = [k for k, v in data.items() if v == min_value]

            min_results[axis] = min_keys
            max_results[axis] = max_keys

        return min_results, max_results

    def calculate_min_max_configurations(self, hists: dict):
        """
        Calculates the minimum and maximum configurations for each axis based on the histograms.
        @param hists:
        @return:
        """
        min_results = {}
        max_results = {}
        for axis, data in hists.items():
            max_value = max(data.values())
            min_value = min(data.values())

            # Check for ties in max and min values
            max_keys = [k for k, v in data.items() if v == max_value]
            min_keys = [k for k, v in data.items() if v == min_value]

            # Decide what to append based on whether there are ties
            if len(max_keys) == 1:
                best_selection = max_keys[0]
            else:
                best_selection = None  # More than one key has the max value or all values are equal

            if len(min_keys) == 1:
                worst_selection = min_keys[0]
            else:
                worst_selection = None  # More than one key has the min value or all values are equal

            min_results[axis] = worst_selection
            max_results[axis] = best_selection

        return min_results, max_results

    def read_group_of_templates(self, model, template_name, datasets: List[str]) -> dict:
        datasets_to_groups = {}
        for dataset in datasets:
            template_groups = self.read_group_of_template(model, dataset)
            template_groups = template_groups[['statistic, pvalue, row is better than column', 'group']]
            # change the value of the cell in the 'statistic, pvalue, row is better than column' if it is contains "best set:" rmpve
            # it and save the name of the template itself
            template_groups['statistic, pvalue, row is better than column'] = template_groups[
                'statistic, pvalue, row is better than column'].map(
                lambda x: x.replace("best set: ", "") if "best set: " in x else x)
            # change the index of the dataframe to be the name of the template
            result = template_groups.set_index('statistic, pvalue, row is better than column')
            # take the index of the row equals to the template_name
            chosen_row = result.loc[template_name]
            chosen_group = chosen_row[ResultConstants.GROUP]

            # count the percentage of the group in the dataset
            group_percentage = round(template_groups[template_groups[ResultConstants.GROUP] == chosen_group].shape[0] / \
                                     template_groups.shape[0], 2)
            datasets_to_groups[dataset] = chosen_group, group_percentage

        return datasets_to_groups

    def read_group_of_template(self, model, dataset):
        results_folder = self.file_path.parent
        groups_path = results_folder / model / dataset / \
                      Path(ResultConstants.ZERO_SHOT) / \
                      Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                      Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
        template_groups = pd.read_csv(groups_path)
        return template_groups

    def check_group_of_conf(self, conf: dict, model: str, datasets: List[str]) -> None:
        """
        Checks the group of the configuration.
        @param conf:
        @param model:
        @param group:
        @return:
        """
        # 1. need to read metadata templates to get the template of the this confiuration
        # 2. need to read the file of the groups of templates, and find the group of this template, that is the group of the configuration
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / TemplatesGeneratorConstants.TEMPLATES_METADATA
        templates_metadata = pd.read_csv(metadata_file, index_col='template_name')

        def generate_combinations(options_dict):
            return [dict(zip(options_dict, values)) for values in itertools.product(*options_dict.values())]

        combinations = generate_combinations(conf)
        for i, combination in enumerate(combinations):
            templates_metadata_mask = templates_metadata.apply(
                lambda x: all([x[key] == value for key, value in combination.items()]), axis=1)
            template = templates_metadata[templates_metadata_mask]
            if len(template) == 0:
                return 'None'
            template_name = template.index[0]

            datasets_to_groups = self.read_group_of_templates(model, template_name, datasets=datasets)
            # calculate the statistics of the groups (how many times each group appears)

            groups_data = list(datasets_to_groups.values())
            groups = [group for group, _ in groups_data]
            percentages_of_chosen_group_in_each_dataset = [percentage for _, percentage in groups_data]
            # split percentages_of_chosen_group_in_each_dataset to groups based on the group value
            groups_percentages = {}
            for group, percentage in zip(groups, percentages_of_chosen_group_in_each_dataset):
                if group not in groups_percentages:
                    groups_percentages[group] = []
                groups_percentages[group].append(percentage)
            # calculate the average of the percentages of the group in the datasets
            groups_percentages = {group: np.median(percentages) for group, percentages in groups_percentages.items()}
            # sort the groups by the percentage
            groups_statistics = Counter(groups)
            # calculate the percentage of the groups in the len of the datasets
            groups_percentage = {group: count / len(datasets) for group, count in groups_statistics.items()}
            # sort the groups by the percentage
            groups_percentage = dict(sorted(groups_percentage.items(), key=lambda item: item[1], reverse=True))
            # print the groups and the percentage tp streamlit
            # st.write(f"Groups of the configuration: {conf}")
            conf_title = f"Configuration {list(combination.values())} is in the following groups:"
            st.markdown(f'<span style="color:red ; font-size: 16px;">{conf_title}</span>', unsafe_allow_html=True)
            for group, percentage in groups_percentage.items():
                st.write(f"In Group: {group} in {(percentage * 100.):.2f}% of the datasets (the median "
                         f" size of the group in the datasets is {(groups_percentages[group] * 100.):.2f}%)")
            st.write("")

    def get_dfs_of_the_cur_group(self, model, group_or_top_k, top_k, dataset_names):
        dfs = []
        # ask to use if take the best k or the first group, if the top_k ask for the k


        for dataset_name in dataset_names:
            results_folder = self.file_path.parent
            groups_path = results_folder / model / dataset_name / \
                          Path(ResultConstants.ZERO_SHOT) / \
                          Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                          Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
            if not Path(groups_path).exists():
                continue
            template_groups_df = pd.read_csv(groups_path)
            if group_or_top_k == 'First group':
                groups = template_groups_df['group'].values
                # take that last lexically group
                groups = sorted(groups)
                last_group = groups[-1]
                template_groups_df = template_groups_df[template_groups_df['group'] == last_group]
            else:
                # take the top k by the accuracy column
                # sorted_df = template_groups_df.sort_values(by='accuracy', ascending=True)

                # Get the maximum accuracy value
                # max_accuracy = sorted_df['accuracy'].max()
                #
                # # Filter rows that have the maximum accuracy
                # top_accuracy_df = sorted_df[sorted_df['accuracy'] == max_accuracy]
                #
                # # Randomly select one row from those with the highest accuracy
                # random_top_accuracy_row = top_accuracy_df.sample(n=top_k)
                # # template_groups_df = template_groups_df.sort_values(by='accuracy', ascending=False).head(top_k)
                # template_groups_df = random_top_accuracy_row
                template_groups_df = template_groups_df.sort_values(by='accuracy', ascending=True).head(top_k)

            template_groups_df['statistic, pvalue, row is better than column'] = template_groups_df[
                'statistic, pvalue, row is better than column'].map(
                lambda x: x.replace("best set: ", "") if "best set: " in x else x)
            # also replace the columns that contain the string "best set: " in the prefix
            template_groups_df.columns = template_groups_df.columns.map(
                lambda x: x.replace("best set: ", "") if "best set: " in x else x)
            # set index to the be ('statistic, pvalue, row is better than column')
            template_groups_df.set_index('statistic, pvalue, row is better than column', inplace=True)
            dfs.append(template_groups_df)
        return dfs

    def get_template_to_conf(self, dfs_cur_group):
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / TemplatesGeneratorConstants.TEMPLATES_METADATA
        templates_metadata = pd.read_csv(metadata_file, index_col='template_name')
        template_to_conf = []
        for df in dfs_cur_group:
            template_cols = [col for col in df if 'template' in col]
            for template_col in template_cols:
                template_to_conf.append(templates_metadata.loc[template_col].to_dict())
        return pd.DataFrame(template_to_conf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=BEST_COMBINATIONS_PATH,
                        help="Path to the best combinations file")
    args = parser.parse_args()
    best_combinations_displayer = BestCombinationsDisplayer(args.file_path, ConfigParams.override_options)
    best_combinations_displayer.evaluate()
