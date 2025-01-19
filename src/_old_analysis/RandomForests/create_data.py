import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.Constants import Constants
from src.utils.MMLUData import MMLUData

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
BestCombinationsConstants = Constants.BestCombinationsConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
MMLUConstants = Constants.MMLUConstants

RESULT_FOLDER = (Path(ExperimentConstants.MAIN_RESULTS_PATH) /
                 TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME)


class RandomForest:
    def __init__(self, result_folder: Path):
        self.result_folder: Path = result_folder
        self.best_combinations_file_path = self.result_folder / Path(f"{ResultConstants.BEST_COMBINATIONS}.csv")
        self.best_combinations: pd.DataFrame = pd.read_csv(self.best_combinations_file_path)
        self.best_combinations = self.best_combinations[
            self.best_combinations[BestCombinationsConstants.DATASET].str.startswith(MMLUConstants.MMLU_CARDS_PREFIX)]

    def read_group_of_template(self, model, dataset):
        groups_path = self.result_folder / model / dataset / \
                      Path(ResultConstants.ZERO_SHOT) / \
                      Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                      Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
        if not groups_path.exists():
            return None
        template_groups = pd.read_csv(groups_path)
        template_groups = self.post_process_template_groups(template_groups, model, dataset)
        return template_groups

    def convert_templates_to_configurations(self, dfs_of_the_cur_group: List[pd.DataFrame]) -> pd.DataFrame:
        templates_metadata = pd.read_csv(TemplatesGeneratorConstants.TEMPLATES_METADATA_PATH, index_col='template_name')
        confs_from_templates = []
        for df in dfs_of_the_cur_group:
            template_cols = df.index.values.tolist()

            for template_col in template_cols:
                axes = templates_metadata.loc[template_col].to_dict()
                # added the axes dict to the df to be able to use it in the display,
                # each key in the dict is a column and the value is the value of the column
                for k, v in axes.items():
                    df.loc[template_col, k] = v
            confs_from_templates.append(df)
        return pd.concat(confs_from_templates)

    def evaluate(self) -> None:
        """
        Renders the evaluation of the best combinations.

        Displays the best combinations and allows the user to split by model or dataset.
        """
        models = self.best_combinations[BestCombinationsConstants.MODEL].unique()
        dfs = []
        for model in models:
            for dataset_name in self.best_combinations[BestCombinationsConstants.DATASET].unique():
                template_groups_df = self.read_group_of_template(model, dataset_name)
                if template_groups_df is None:
                    continue
                dfs.append(template_groups_df)
        # concatenate the dfs to new df
        configurations_data = self.convert_templates_to_configurations(dfs)
        # save the configurations data to a csv file in the results folder
        configurations_data.to_csv(Path(__file__).parent / "configurations_data.csv")

    def _add_mmlu_columns(self, model_data: pd.DataFrame) -> None:
        """
        Adds MMLU subcategories and categories columns to the model data.

        @param model_data: DataFrame containing model data to update.
        """
        MMLUData.initialize()
        MMLUData._add_mmlu_columns(model_data)

    def post_process_template_groups(self, dataset_configurations_groups, model, dataset):
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
        result.index.name = 'template_name'

        # add the model and dataset to the dataframe in new columns
        result['model'] = model
        result['dataset'] = dataset
        # added mmlu columns
        self._add_mmlu_columns(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default=RESULT_FOLDER,
                        help="Path to the best combinations file")
    args = parser.parse_args()
    rf = RandomForest(args.result_folder)
    rf.evaluate()
