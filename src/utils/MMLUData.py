from pathlib import Path

import pandas as pd

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
BestCombinationsConstants = Constants.BestCombinationsConstants
MMLUConstants = Constants.MMLUConstants


class MMLUData:
    mmlu_metadata_file = None
    mmlu_metadata = None

    @classmethod
    def initialize(cls):
        cls.mmlu_metadata_file = Path(__file__).parents[2] / TemplatesGeneratorConstants.MMLU_DATASET_SIZES_PATH
        cls.mmlu_metadata = pd.read_csv(cls.mmlu_metadata_file)

    @staticmethod
    def add_mmlu_prefix(dataset_name):
        return f'{MMLUConstants.MMLU_CARDS_PREFIX}{dataset_name}'

    @staticmethod
    def get_mmlu_dataset_sizes():
        return MMLUData.mmlu_metadata

    @staticmethod
    def get_mmlu_categories():
        return MMLUData.mmlu_metadata[MMLUConstants.CATEGORIES_COLUMN].unique().tolist()

    @staticmethod
    def get_mmlu_subcategories():
        return MMLUData.mmlu_metadata[MMLUConstants.SUBCATEGORIES_COLUMN].unique().tolist()

    @staticmethod
    def get_mmlu_subcategories_from_category(category):
        return MMLUData.mmlu_metadata[MMLUData.mmlu_metadata[MMLUConstants.CATEGORIES_COLUMN] == category][
            MMLUConstants.SUBCATEGORIES_COLUMN].unique()

    @staticmethod
    def get_mmlu_datasets_from_subcategory(subcategory):
        return (MMLUData.mmlu_metadata[MMLUData.mmlu_metadata[MMLUConstants.SUBCATEGORIES_COLUMN] == subcategory][
                    MMLUConstants.ALL_DATASETS_COLUMN]
                .apply(lambda x: MMLUData.add_mmlu_prefix(x)).unique())

    @staticmethod
    def get_mmlu_datasets():
        mmlu_datasets = [f'{MMLUConstants.MMLU_CARDS_PREFIX}{mmlu_dataset}' for mmlu_dataset in
                         MMLUData.mmlu_metadata[MMLUConstants.ALL_DATASETS_COLUMN]]
        return mmlu_datasets

    @staticmethod
    def _add_mmlu_columns(model_data: pd.DataFrame) -> None:
        """
        Adds MMLU subcategories and categories columns to the model data.

        @param model_data: DataFrame containing model data to update.
        """
        mmlu_dataset = MMLUData.get_mmlu_dataset_sizes()
        # add a column MMLUConstants.SUBCATEGORIES_COLUMN to the model_data DataFrame, and the values will be the
        # values from mmlu_dataset by the dataset name - so take the row from mmlu_dataset that the dataset name is
        # the same as the value in the column BestCombinationsConstants.DATASET in the model_data DataFrame, and take
        # the value from the column MMLUConstants.SUBCATEGORIES_COLUMN in the mmlu_dataset DataFrame
        mmlu_dataset.set_index(MMLUConstants.ALL_DATASETS_COLUMN, inplace=True)
        model_data[MMLUConstants.SUBCATEGORIES_COLUMN] = model_data[BestCombinationsConstants.DATASET].apply(
            lambda x: mmlu_dataset.loc[x.split(MMLUConstants.MMLU_CARDS_PREFIX)[1], MMLUConstants.SUBCATEGORIES_COLUMN])
        model_data[MMLUConstants.CATEGORIES_COLUMN] = model_data[BestCombinationsConstants.DATASET].apply(
            lambda x: mmlu_dataset.loc[x.split(MMLUConstants.MMLU_CARDS_PREFIX)[1], MMLUConstants.CATEGORIES_COLUMN])
