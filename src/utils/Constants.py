from pathlib import Path


class Constants:
    class DatasetModifier:
        DATA_PATH = Path(__file__).parents[1] / "Data"
        MODIFIED_DATA_PATH = Path(__file__).parents[1] / "ModifiedData"
        TEST_FILE = "test.json"

    class SocialQAModifier:
        DATA_NAME = "SocialQA"

    class WinograndeModifier:
        DATA_NAME = "Winogrande"

    class MMLUModifier:
        DATA_NAME = "MMLU"
