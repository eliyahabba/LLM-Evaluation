from pathlib import Path


class Constants:
    class UnitxtDataConstants:
        DATA_PATH = 'unitxt/data'
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME
