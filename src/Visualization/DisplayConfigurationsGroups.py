import itertools
from collections import Counter
from typing import List

import numpy as np
import streamlit as st

from src.Visualization.GetGroupOfConfiguration import GetGroupOfConfiguration
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants


class DisplayConfigurationsGroups:
    def __init__(self, model_results_path, templates_metadata):
        self.model_results_path = model_results_path
        self.templates_metadata = templates_metadata

    def check_the_group_of_conf(self, configuration: dict, datasets: List[str], num_of_expected_datasets: int, num_od_actual_datasets: int) -> None:
        """
        Checks the group of the configuration.
        @param configuration:
        @param datasets:
        @param num_of_expected_datasets:
        @param num_od_actual_datasets:
        @return:
        """
        st.markdown(f"Coverage of: {num_od_actual_datasets}/{num_of_expected_datasets} datasets")
        # 1. need to read metadata templates to get the template of the this confiuration
        # 2. need to read the file of the groups of templates, and find the group of this template, that is the group of the configuration

        def generate_combinations(options_dict):
            return [dict(zip(options_dict, values)) for values in itertools.product(*options_dict.values())]

        combinations = generate_combinations(configuration)
        for i, combination in enumerate(combinations):
            templates_metadata_mask = self.templates_metadata.apply(
                lambda x: all([x[key] == value for key, value in combination.items()]), axis=1)

            chosen_template_name = self.templates_metadata[templates_metadata_mask].index[0]
            get_group_of_configuration = GetGroupOfConfiguration(self.model_results_path)
            chosen_groups_with_percentages = get_group_of_configuration.read_group_of_templates(chosen_template_name,
                                                                                                datasets=datasets)
            # calculate the statistics of the groups (how many times each group appears)

            chosen_groups = [group for group, _ in chosen_groups_with_percentages]
            percentages_of_chosen_group_in_each_dataset = [percentage for _, percentage in
                                                           chosen_groups_with_percentages]

            # split percentages_of_chosen_group_in_each_dataset to groups based on the group value
            groups_percentages = {}
            for group, percentage in zip(chosen_groups, percentages_of_chosen_group_in_each_dataset):
                if group not in groups_percentages:
                    groups_percentages[group] = []
                groups_percentages[group].append(percentage)
            # calculate the average of the percentages of the group in the datasets
            groups_percentages = {group: np.median(percentages) for group, percentages in groups_percentages.items()}
            # sort the groups by the percentage
            groups_statistics = Counter(chosen_groups)
            # calculate the percentage of the groups in the len of the datasets
            groups_percentage = {group: count / len(datasets) for group, count in groups_statistics.items()}
            # sort the groups by the percentage
            groups_percentage = dict(sorted(groups_percentage.items(), key=lambda item: item[1], reverse=True))
            # print the groups and the percentage tp streamlit
            self.display_configuration_stats(combination, groups_percentage, groups_percentages)

    def display_configuration_stats(self, combination: dict, groups_percentage: dict, groups_percentages: dict):

        conf_title = f"Configuration {list(combination.values())} is in the following groups:"
        st.markdown(f'<span style="color:red ; font-size: 16px;">{conf_title}</span>', unsafe_allow_html=True)
        for group, percentage in groups_percentage.items():
            st.write(f"In Group: {group} in {(percentage * 100.):.2f}% of the datasets (the median "
                     f" size of the group in the datasets is {(groups_percentages[group] * 100.):.2f}%)")
        st.write("")
