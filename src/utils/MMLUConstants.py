from pathlib import Path

import pandas as pd

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
class MMLUConstants:
    def __init__(self):
        mmlu_metadata_file =  Path(__file__).parents[2] / TemplatesGeneratorConstants.MMLU_DATASET_SIZES_PATH
        self.mmlu_metadata = pd.read_csv(mmlu_metadata_file)
    def get_mmlu_dataset_sizes(self):
        return self.mmlu_metadata

    def get_mmlu_categories(self):
        return self.mmlu_metadata['category'].unique()

    def get_mmlu_subcategories(self):
        return self.mmlu_metadata['subcategory'].unique()

    def get_mmlu_

    MMLU_DATASETS_SAMPLE = [f'mmlu.{mmlu_dataset}' for mmlu_dataset in SUBCATEGORIES.keys()]

    CATEGORIES_COLUMN = "categories"
    SUBCATEGORIES_COLUMN = "subcategories"
    ALL_DATASETS_COLUMN = "all datasets"
    SPLIT_OPTIONS = [CATEGORIES_COLUMN, SUBCATEGORIES_COLUMN, ALL_DATASETS_COLUMN]

    MMLU_NAME = "mmlu"
    MMLU_CARDS_PREFIX = f"{MMLU_NAME}."