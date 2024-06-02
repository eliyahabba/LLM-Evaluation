from pathlib import Path

import pandas as pd

from src.utils.Constants import Constants
from src.utils.MMLUData import MMLUData

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants

class DatasetsManger:
    SCIQ = "sciq"
    RACE_ALL = "race_all"
    AI2_ARC_ARC_EASY = "ai2_arc.arc_easy"
    HELLASWAG = "hellaswag"
    BASE_DATASET_NAMES = [SCIQ, RACE_ALL, AI2_ARC_ARC_EASY]
    MMLUData.initialize()
    DATASET_NAMES = BASE_DATASET_NAMES + MMLUData.get_mmlu_datasets()

    @staticmethod
    def get_dataset_names():
        return DatasetsManger.DATASET_NAMES

    @staticmethod
    def get_base_dataset_names():
        return DatasetsManger.BASE_DATASET_NAMES