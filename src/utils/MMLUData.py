from pathlib import Path

import pandas as pd

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class MMLUData:
    mmlu_metadata_file = None
    mmlu_metadata = None

    @classmethod
    def initialize(cls):
        cls.mmlu_metadata_file = Path(__file__).parents[2] / TemplatesGeneratorConstants.MMLU_DATASET_SIZES_PATH
        cls.mmlu_metadata = pd.read_csv(cls.mmlu_metadata_file)

    @staticmethod
    def add_mmlu_prefix(dataset_name):
        return f'mmlu.{dataset_name}'

    @staticmethod
    def get_mmlu_dataset_sizes():
        return MMLUData.mmlu_metadata

    @staticmethod
    def get_mmlu_categories():
        return MMLUData.mmlu_metadata['Category'].unique().tolist()

    @staticmethod
    def get_mmlu_subcategories():
        return MMLUData.mmlu_metadata['Sub_Category'].unique().tolist()

    @staticmethod
    def get_mmlu_subcategories_from_category(category):
        return MMLUData.mmlu_metadata[MMLUData.mmlu_metadata['Category'] == category]['Sub_Category'].unique()

    @staticmethod
    def get_mmlu_datasets_from_subcategory(subcategory):
        return (MMLUData.mmlu_metadata[MMLUData.mmlu_metadata['Sub_Category'] == subcategory]['Name']
                .apply(lambda x: MMLUData.add_mmlu_prefix(x)).unique())

    @staticmethod
    def get_mmlu_datasets():
        mmlu_datasets = [f'mmlu.{mmlu_dataset}' for mmlu_dataset in MMLUData.mmlu_metadata['Name']]
        return mmlu_datasets