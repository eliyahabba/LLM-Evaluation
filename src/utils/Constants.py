from pathlib import Path


class Constants:
    class TemplatesGeneratorConstants:
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME

    class ExperimentConstants:
        RESULTS_FOLDER_NAME = "results"
        RESULTS_PATH = Path(__file__).parents[2] / RESULTS_FOLDER_NAME
