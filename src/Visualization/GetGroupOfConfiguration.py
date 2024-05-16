from pathlib import Path
from typing import List

import pandas as pd

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants


class GetGroupOfConfiguration:
    def __init__(self, results_folder, model):
        self.results_folder = results_folder
        self.model = model

    def read_group_of_templates(self, template_name, datasets: List[str]) -> list:
        datasets_to_groups = {}
        for dataset in datasets:
            dataset_configurations_groups = self.read_group_of_template(dataset)
            dataset_configurations_groups = dataset_configurations_groups[
                ['statistic, pvalue, row is better than column', 'group']]
            # change the value of the cell in the 'statistic, pvalue, row is better than column' if it is contains "best set:" rmpve
            # it and save the name of the template itself
            dataset_configurations_groups['statistic, pvalue, row is better than column'] = \
                dataset_configurations_groups[
                    'statistic, pvalue, row is better than column'].map(
                    lambda x: x.replace("best set: ", "") if "best set: " in x else x)
            # change the index of the dataframe to be the name of the template
            result = dataset_configurations_groups.set_index('statistic, pvalue, row is better than column')
            # take the index of the row equals to the template_name
            chosen_row = result.loc[template_name]
            chosen_group = chosen_row[ResultConstants.GROUP]

            # count the percentage of the group in the dataset
            group_percentage = round(dataset_configurations_groups[
                                         dataset_configurations_groups[ResultConstants.GROUP] == chosen_group].shape[
                                         0] / \
                                     dataset_configurations_groups.shape[0], 2)
            datasets_to_groups[dataset] = chosen_group, group_percentage

        return list(datasets_to_groups.values())

    def read_group_of_template(self, dataset):
        groups_path = self.results_folder / self.model / dataset / \
                      Path(ResultConstants.ZERO_SHOT) / \
                      Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                      Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
        template_groups = pd.read_csv(groups_path)
        return template_groups
