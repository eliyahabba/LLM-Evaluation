import itertools
import re
import sys
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.DataProcessing.MMLUSplitter import MMLUSplitter
from src.RandomForests.Constants import RandomForestsConstants
from src.RandomForests.random_forest import RandomForest
from src.utils.MMLUData import MMLUData

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.Visualization.DisplayConfigurationsGroups import DisplayConfigurationsGroups
from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
BestCombinationsConstants = Constants.BestCombinationsConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
BEST_COMBINATIONS_PATH = ExperimentConstants.MAIN_RESULTS_PATH / Path(
    TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME) / f"{ResultConstants.BEST_COMBINATIONS}.csv"
MMLUConstants = Constants.MMLUConstants
BestOrWorst = Constants.BestOrWorst
MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH


class FindCombinations:
    def __init__(self, best_or_worst: BestOrWorst, override_options: dict = None, families: bool = False) -> None:
        """
        Initialize the BestCombinationsDisplayer class.

        @param file_path: Path to the CSV file containing the best combinations.
        @param override_options: Dictionary containing options to override histograms.
        """
        self.best_combinations = None
        self.best_combinations_file_path = None
        self.results_folder = None
        self.model_results_path = None

        self.best_or_worst = best_or_worst
        self.main_results_path = Path(MAIN_RESULTS_PATH)
        self.override_options: dict = override_options
        self.families = families
        self._read_templates_metadata()

    def get_result_files(self) -> Path:
        """
        Get the result files.
        @return:
        """
        # ask the use to select the results folder from the ExperimentConstants.MAIN_RESULTS_PATH
        results_folders = [folder for folder in self.main_results_path.iterdir() if folder.is_dir()]
        # take the part name of the folder
        results_folders = [folder.parts[-1] for folder in results_folders]
        # split each name to the last word that start with capital letter
        results_names = [re.findall('[A-Z][^A-Z]*', folder)[-1] for folder in results_folders]
        # join the words with space
        chosen_results_names = st.selectbox("Select the template prompt of the results", results_names)
        results_folder_ind = results_names.index(chosen_results_names)
        return self.main_results_path / Path(results_folders[results_folder_ind])

    def read_best_combinations_file(self) -> pd.DataFrame:
        """
        Reads the best combinations DataFrame.

        @return: DataFrame containing the best combinations.
        """
        return pd.read_csv(self.best_combinations_file_path)

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
        MMLUData.initialize()
        MMLUData._add_mmlu_columns(model_data=model_data)

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

    def _create_histograms_plots(self, cur_data: pd.DataFrame) -> dict:
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

    def choose_group_or_top_k(self, model_name: str) -> Tuple[str, int]:
        group_or_top_k = st.radio("Do you want to take the best k or the first group?", ['First group', 'Top k'],
                                  key=model_name)
        top_k = None
        if group_or_top_k == 'Top k':
            top_k = st.number_input("Enter the number of top k groups you want to take", 1, 20, 1, key=model_name)

        return group_or_top_k, top_k

    def evaluate_the_model_in_family(self, model_name: str, model_data: pd.DataFrame,
                                     display_histograms: bool = False,
                                     split_option: str = None,
                                     display_full_details: bool = False, i: int = None,
                                     first_model: bool = False) -> None:

        model_data_splitted = MMLUSplitter.split_data_by_option(model_data, split_option)
        group_or_top_k = 'First group'
        top_k = None
        # self.choose_group_or_top_k(model_name)
        display_configurations_groups = DisplayConfigurationsGroups(self.model_results_path, self.templates_metadata,
                                                                    display_full_details=display_full_details,
                                                                    first_model=first_model)
        # take the i element from the model_data_splitted (dict)
        group_name, cur_data = list(model_data_splitted.items())[i]
        # read the df of the current group
        dataset_names = cur_data.dataset.values
        datasets_of_the_current_group = self.get_datasets_of_the_current_group(group_or_top_k, top_k,
                                                                               dataset_names)
        # concatenate the dfs to new df
        best_templates_of_cur_group = [df.index.values.tolist() for df in datasets_of_the_current_group]
        # map between the name of template and configuration
        selected_configurations_df = self.convert_templates_names_to_conf_values(best_templates_of_cur_group)
        configurations_counter = self._create_histograms_plots(selected_configurations_df)
        if first_model:
            st.markdown(f'<span style="font-size: 20px; color:blue">{group_name}</span>', unsafe_allow_html=True)

        if display_histograms:
            self.display_bars(configurations_counter, group_name, cur_data)

        most_common_configuration = self.calculate_most_common_configurations(configurations_counter)
        # if first_model:
        #     st.markdown(f'<span style="font-size: 17px; color:green">**The {self.best_or_worst.value} configurations of {model_name}**</span>',
        #                 unsafe_allow_html=True)
        num_of_expected_datasets = len(dataset_names)
        num_od_actual_datasets = len(datasets_of_the_current_group)
        st.markdown(
            f'<span style="font-size: 16px; color:black">{model_name}:</span><span style="font-size: 16px; color:gray"> Coverage of: {num_od_actual_datasets}/{num_of_expected_datasets} datasets</span>',
            unsafe_allow_html=True)
        display_configurations_groups.check_the_group_of_conf(most_common_configuration, cur_data.dataset.values)
        # add empty line
        st.write("")

    def evaluate_the_model(self, model_name: str, model_data: pd.DataFrame,
                           display_histograms: bool = False,
                           split_option: str = None,
                           display_full_details: bool = True) -> None:
        """
        Evaluates data by the specified key (model or dataset) and splits if required.

        @param model_data: DataFrame containing model or dataset data.
        @param split_option: Option to split data into subcategories or categories.
        """
        model_data_splitted = self.split_data_by_option(model_data, split_option)
        group_or_top_k, top_k = self.choose_group_or_top_k(model_name)
        display_configurations_groups = DisplayConfigurationsGroups(self.model_results_path, self.templates_metadata,
                                                                    display_full_details=display_full_details)
        st.markdown(
            f'<span style="font-size: 17px; color:green">**The {self.best_or_worst.value} configurations of {model_name}**</span>',
            unsafe_allow_html=True)

        for group_name, cur_data in model_data_splitted.items():
            # read the df of the current group
            dataset_names = cur_data.dataset.values
            datasets_of_the_current_group = self.get_datasets_of_the_current_group(group_or_top_k, top_k,
                                                                                   dataset_names)
            # concatenate the dfs to new df
            best_templates_of_cur_group = [df.index.values.tolist() for df in datasets_of_the_current_group]
            # map between the name of template and configuration
            selected_configurations_df = self.convert_templates_names_to_conf_values(best_templates_of_cur_group)
            configurations_counter = self._create_histograms_plots(selected_configurations_df)
            st.markdown(f'<span style="font-size: 20px; color:blue">{group_name}</span>', unsafe_allow_html=True)

            if display_histograms:
                self.display_bars(configurations_counter, group_name, cur_data)

            most_common_configuration = self.calculate_most_common_configurations(configurations_counter)

            num_of_expected_datasets = len(dataset_names)
            num_od_actual_datasets = len(datasets_of_the_current_group)
            st.markdown(f"Coverage of: {num_od_actual_datasets}/{num_of_expected_datasets} datasets")
            display_configurations_groups.check_the_group_of_conf(most_common_configuration, cur_data.dataset.values)
            # add empty line
            self.predict_with_random_forest(most_common_configuration, model_name, group_name, cur_data)
            st.write("")

    def predict_with_random_forest(self, most_common_configuration, model_name: str, group_name: str,
                                   cur_data: pd.DataFrame) -> None:
        """
        Predicts the best configurations for the specified model using Random Forest.

        @param model_name: Name of the model to predict the best configurations for.
        """
        rf = RandomForest(feature_columns=["dataset", "Sub_Category", "Category", "enumerator", "choices_separator", "shuffle_choices"])
        rf.load_data(model=model_name)
        rf.create_model()
        X_train, X_test, y_train, y_test = rf.split_data(split_column_name=RandomForestsConstants.CATEGORY,
                                                         test_column_values=[group_name])
        rf.train(X_train, y_train)

        # predictions = rf.predict(X_test)
        # metrics = rf.evaluate(y_test, predictions, print_metrics=True)
        def generate_combinations(options_dict):
            return [dict(zip(options_dict, values)) for values in itertools.product(*options_dict.values())]

        combinations = generate_combinations(most_common_configuration)
        predictions = []
        for i, combination in enumerate(combinations):
            combination = self.add_metadata_to_combination(combination, cur_data)
            prediction = rf.predict(pd.DataFrame(combination, index=[0]))
            # this only one prediction so we can take the first one
            prediction = prediction[0]
            predictions.append(prediction)

        results_str = '<span style="font-size: 17px; color:orange">**Predicting the groups of the configurations of this subset of datasets using Random Forest predict these configurations for the group:**</span> ' + ' '.join(
            f'<span style="color:black">**{result}**</span>' for result in predictions)
        st.markdown(results_str, unsafe_allow_html=True)

    def display_bars(self, hists: dict, group: str, cur_data: pd.DataFrame) -> None:
        figs = self._create_figure(hists)
        st.write(f"Group: {group}, Number of samples: {len(cur_data)}")
        cols = st.columns(len(figs))
        for i, fig in enumerate(figs):
            with cols[i]:
                st.pyplot(fig)

    def read_best_combinations(self) -> None:
        self.results_folder = self.get_result_files()
        self.best_combinations_file_path = self.results_folder / Path(f"{ResultConstants.BEST_COMBINATIONS}.csv")
        self.best_combinations: pd.DataFrame = self.read_best_combinations_file()
        self._filter_best_combinations()

    def evaluate(self) -> None:
        """
        Renders the evaluation of the best combinations.

        Displays the best combinations and allows the user to split by model or dataset.
        """
        st.title("Evaluation by model")
        display_histograms = st.checkbox("Display histograms")

        self.read_best_combinations()
        if self.families:
            families = LLMProcessorConstants.MODELS_FAMILIES.keys()
            family = st.selectbox("Choose a family", families)
            self.display_family_results(family, display_histograms)
        else:
            models = self.best_combinations[BestCombinationsConstants.MODEL].unique()
            model = st.selectbox("Choose a model", models)
            self.display_model_results(model, display_histograms)

    def display_family_results(self, family: str, display_histograms: bool) -> None:
        """
        Displays the results of the family.
        @param family:
        @param display_histograms:
        @return:
        """
        fam = LLMProcessorConstants.MODELS_FAMILIES[family]
        split_option = st.selectbox("Split the dataset by:", MMLUConstants.SPLIT_OPTIONS)
        st.markdown(
            f'<span style="font-size: 17px; color:green">**The {self.best_or_worst.value} configurations of:**</span>',
            unsafe_allow_html=True)
        for model in fam:
            st.markdown(f'<span style="font-size: 17px; color:green">{model}</span>',
                        unsafe_allow_html=True)

        for i in range(4):
            for j, full_model_name in enumerate(LLMProcessorConstants.MODELS_FAMILIES[family]):
                model_name = full_model_name.split("/")[1]
                self.model_results_path = self.results_folder / model_name

                model_data = self.best_combinations[
                    self.best_combinations[BestCombinationsConstants.MODEL] == model_name]
                model_data = model_data.sort_values(by=BestCombinationsConstants.DATASET).reset_index(drop=True)
                self._add_mmlu_columns(model_data)
                self.evaluate_the_model_in_family(model_name, model_data, display_histograms, split_option, i=i,
                                                  first_model=True
                                                  if j == 0 else False)

    def display_model_results(self, model: str, display_histograms: bool) -> None:
        """
        Displays the results of the model.
        @param model:
        @param display_histograms:
        @return:
        """

        self.model_results_path = self.results_folder / model
        model_data = self.best_combinations[self.best_combinations[BestCombinationsConstants.MODEL] == model]
        model_data = model_data.sort_values(by=BestCombinationsConstants.DATASET).reset_index(drop=True)
        self._add_mmlu_columns(model_data)

        split_option = st.selectbox("Split the dataset by:", MMLUConstants.SPLIT_OPTIONS)

        self.evaluate_the_model(model, model_data, display_histograms, split_option, display_full_details=True)

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

    def get_datasets_of_the_current_group(self, group_or_top_k: str, top_k: Union[int, None], dataset_names: List[str]):
        dfs = []

        for dataset_name in dataset_names:
            groups_path = self.model_results_path / dataset_name / \
                          Path(ResultConstants.ZERO_SHOT) / \
                          Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                          Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
            if not groups_path.exists():
                continue
            template_groups_df = pd.read_csv(groups_path)

            reverse_and_ascending = False if self.best_or_worst == BestOrWorst.BEST else True
            if group_or_top_k == 'First group':
                groups = template_groups_df['group'].values
                # remove nan values
                groups = [group for group in groups if str(group) != 'nan']
                # take that last lexically group
                groups = sorted(groups, reverse=reverse_and_ascending)
                last_group = groups[0]
                template_groups_df = template_groups_df[template_groups_df['group'] == last_group]
            else:
                template_groups_df = template_groups_df.sort_values(by='accuracy',
                                                                    ascending=reverse_and_ascending).head(top_k)
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

    def add_metadata_to_combination(self, combination:dict, cur_data: pd.DataFrame)->dict:
        """
        Adds metadata to the combination.
        @param combination:
        @param cur_data:
        @return:
        """
        combination["dataset"] = cur_data["dataset"].values[0]
        combination["Sub_Category"] = cur_data["Sub_Category"].values[0]
        combination["Category"] = cur_data["Category"].values[0]
        return combination