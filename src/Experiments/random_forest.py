# Usage
import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.MMLUConstants import MMLUConstants

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
BestCombinationsConstants = Constants.BestCombinationsConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
BEST_COMBINATIONS_PATH = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH / f"{ResultConstants.BEST_COMBINATIONS}.csv"

from src.Experiments.GroupPredictor import GroupPredictor

split_option = MMLUConstants.SPLIT_OPTIONS[0]

MMLU_SUBCATEGORIES = MMLUConstants.SUBCATEGORIES
MMLU_CATEGORIES = MMLUConstants.SUBCATEGORIES_TO_CATEGORIES


class RandomForest:
    def __init__(self, file_path: Path):
        self.file_path: Path = file_path
        self.best_combinations: pd.DataFrame = pd.read_csv(file_path)
        self.data = {}

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

    def _load_data(self,group: str) -> pd.DataFrame:
        self.group_data = self.data[group]

    def create_model(self, group: str):
        self._load_data(group=group)
        self.predictor = GroupPredictor()
        X_train, X_test, y_train, y_test = self.predictor.load_and_split_data(self.data)
        self.predictor.train(X_train, y_train)

    def predict(self):
        # New data for prediction
        new_data = [
            {"model": "Llama-2-7b-chat-hf", "enumerator": "lowercase", "choices_separator": ", ",
             "shuffle_choices": True},
            {"model": "Llama-2-7b-chat-hf", "enumerator": "roman", "choices_separator": " OR ",
             "shuffle_choices": False}
        ]
        new_df = self.predictor.prepare_data(new_data)
        predictions = self.predictor.predict(new_df)
        print("Predictions:", predictions)

    def read_group_of_templates(self, model, datasets: List[str]) -> dict:
        datasets_to_groups = {}
        for dataset in datasets:
            template_group_leader = self.read_group_of_template(model, dataset)
            datasets_to_groups[dataset] = template_group_leader
        return datasets_to_groups

    def read_group_of_template(self, model, dataset):
        results_folder = self.file_path.parent
        groups_path = results_folder / model / dataset / \
                      Path(ResultConstants.ZERO_SHOT) / \
                      Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                      Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
        template_groups = pd.read_csv(groups_path)
        return template_groups

    def evaluate_by_model_or_dataset(self, model: str, model_data: pd.DataFrame, split_option: str = "categories") -> None:
        """
        Evaluates data by the specified key (model or dataset) and splits if required.

        @param model_or_dataset_key: Key to evaluate by (e.g., model).
        @param model_data: DataFrame containing model or dataset data.
        @param split_option: Option to split data into subcategories or categories.
        """
        model_datas = self.split_data_by_option(model_data, split_option)
        top_k = 5
        for group, cur_data in model_datas.items():
            dataset_names = cur_data.dataset.values
            dfs_of_the_cur_group = self.get_dfs_of_the_cur_group(model, dataset_names)
            # concatenate the dfs to new df
            template_to_conf = self.get_template_to_conf(dfs_of_the_cur_group)
            self.data[group] = template_to_conf

    def get_dfs_of_the_cur_group(self, model, dataset_names):
        dfs = []
        # ask to use if take the best k or the first group, if the top_k ask for the k
        for dataset_name in dataset_names:
            results_folder = self.file_path.parent
            groups_path = results_folder / model / dataset_name / \
                          Path(ResultConstants.ZERO_SHOT) / \
                          Path(ResultConstants.EMPTY_SYSTEM_FORMAT) / \
                          Path(ResultConstants.GROUPED_LEADERBOARD + '.csv')
            template_groups_df = pd.read_csv(groups_path)
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

    def get_template_to_conf(self, dfs_of_the_cur_group: List[pd.DataFrame]) -> pd.DataFrame:
        # dfs_cur_group = [df.index.values.tolist() for df in dfs_of_the_cur_group]

        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / TemplatesGeneratorConstants.TEMPLATES_METADATA
        templates_metadata = pd.read_csv(metadata_file, index_col='template_name')
        template_to_conf = []
        for df in dfs_of_the_cur_group:
            df_cols = df.index.values.tolist()
            template_cols = [col for col in df_cols if 'template' in col]
            for template_col in template_cols:
                accuracy = df.loc[template_col]['accuracy']
                group = df.loc[template_col]['group']
                axes = templates_metadata.loc[template_col].to_dict()
                axes['accuracy'] = accuracy
                axes['group'] = group
                template_to_conf.append(axes)
        return pd.DataFrame(template_to_conf)

    def evaluate(self) -> None:
        """
        Renders the evaluation of the best combinations.

        Displays the best combinations and allows the user to split by model or dataset.
        """
        models = self.best_combinations[BestCombinationsConstants.MODEL].unique()
        model = models[0]
        model_data = self.best_combinations[self.best_combinations[BestCombinationsConstants.MODEL] == model]
        model_data = model_data.sort_values(by=BestCombinationsConstants.DATASET).reset_index(drop=True)
        self._add_mmlu_columns(model_data)
        self.evaluate_by_model_or_dataset(model, model_data)
        self.create_model("STEM")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=BEST_COMBINATIONS_PATH,
                        help="Path to the best combinations file")
    args = parser.parse_args()
    rf = RandomForest(args.file_path)
    rf.evaluate()