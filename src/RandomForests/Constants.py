from pathlib import Path


class RandomForestsConstants:
    GROUP = "group"
    CATEGORY = "Category"

    CONFIGURATIONS_DATA_FILE_NAME = "configurations_data.csv"
    CONFIGURATIONS_DATA_PATH = Path(__file__).parents[0] / CONFIGURATIONS_DATA_FILE_NAME
