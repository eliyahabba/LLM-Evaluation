import pandas as pd

from src.utils.Constants import Constants
from src.utils.MMLUData import MMLUData

MMLUConstants = Constants.MMLUConstants


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

    @staticmethod
    def get_data_options(split_option):
        MMLUData.initialize()
        if split_option == MMLUConstants.ALL_NAMES:
            options = []
        elif split_option == MMLUConstants.SUBCATEGORIES_COLUMN:
            options = MMLUData.get_mmlu_subcategories()
        elif split_option == MMLUConstants.CATEGORIES_COLUMN:
            options = MMLUData.get_mmlu_categories()
        else:
            options = MMLUData.get_mmlu_datasets()
        return options

    @staticmethod
    def get_data_files(split_option, value):
        if split_option == MMLUConstants.ALL_NAMES:
            options = []
            categories = MMLUData.get_mmlu_categories()
            for category in categories:
                subs_options = MMLUData.get_mmlu_subcategories_from_category(category)
                for sub_option in subs_options:
                    options.extend(MMLUData.get_mmlu_datasets_from_subcategory(sub_option))
        elif split_option == MMLUConstants.SUBCATEGORIES_COLUMN:
            options = MMLUData.get_mmlu_datasets_from_subcategory(value)
        elif split_option == MMLUConstants.CATEGORIES_COLUMN:
            subs_options = MMLUData.get_mmlu_subcategories_from_category(value)
            options = []
            for sub_option in subs_options:
                options.extend(MMLUData.get_mmlu_datasets_from_subcategory(sub_option))
        else:
            options = [value]
        return options
