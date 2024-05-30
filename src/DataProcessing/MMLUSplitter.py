import pandas as pd

from src.utils.MMLUConstants import MMLUConstants

MMLU_SUBCATEGORIES = MMLUConstants.SUBCATEGORIES
MMLU_CATEGORIES = MMLUConstants.SUBCATEGORIES_TO_CATEGORIES


class MMLUSplitter:
    @staticmethod
    def split_data_by_option(model_data: pd.DataFrame, split_option: str) -> dict:
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

    @classmethod
    def get_data_options(cls, split_option):
        if split_option == MMLUConstants.SUBCATEGORIES_COLUMN:
            all_subcategories = MMLUConstants.CATEGORIES.values()
            options = [subcategories for subcategories in all_subcategories]
        elif split_option == MMLUConstants.CATEGORIES_COLUMN:
            options = list(MMLUConstants.CATEGORIES.keys())
        else:
            options = list(MMLUConstants.SUBCATEGORIES.keys())
        return options

    @classmethod
    def get_data_files(cls, split_option, value):
        if split_option == MMLUConstants.SUBCATEGORIES_COLUMN:
            all_subcategories = MMLUConstants.CATEGORIES.values()
            options = [subcategories for subcategories in all_subcategories]
        elif split_option == MMLUConstants.CATEGORIES_COLUMN:
            options = MMLUConstants.CATEGORIES[value]
            files = []
            for option in options:
                files.extend([subcategories for subcategories in all_subcategories])
        else:
            options = list(MMLUConstants.SUBCATEGORIES.keys())

        return options
