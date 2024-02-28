from pathlib import Path


class Constants:
    class DatasetModifierConstants:
        DATA_PATH = Path(__file__).parents[1] / "Data"
        MODIFIED_DATA_PATH = Path(__file__).parents[1] / "ModifiedData"
        TEST_FILE = "test.json"

    class SocialQAModifierConstants:
        DATA_NAME = "SocialQA"

    class WinograndeModifierConstants:
        DATA_NAME = "Winogrande"

    class MMLUModifierConstants:
        DATA_NAME = "MMLU"
