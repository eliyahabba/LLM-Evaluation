from pathlib import Path


class Constants:
    class DatasetModifierConstants:
        DATA_PATH = Path(__file__).parents[2] / "Data"
        MODIFIED_DATA_PATH = Path(__file__).parents[1] / "ModifiedData"
        TEST_FILE = "test.json"

    class SocialQAModifierConstants:
        DATA_NAME = "SocialQA"

    class WinograndeModifierConstants:
        DATA_NAME = "Winogrande"

    class MMLUModifierConstants:
        DATA_NAME = "MMLU"

    class UnitxtDataConstants:
        DATA_PATH = 'unitxt/data'
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME
