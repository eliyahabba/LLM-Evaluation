from pathlib import Path


class Constants:
    class TemplatesGeneratorConstants:
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME

    class ExperimentConstants:
        RESULTS_FOLDER_NAME = "results"
        RESULTS_PATH = Path(__file__).parents[2] / RESULTS_FOLDER_NAME

        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
        SYSTEM_FORMATS = "formats.empty"
        MAX_INSTANCES = 100
        EVALUATE_ON = ['train', 'test']
        TEMPLATE_NUM = 0
        NUM_DEMOS = 1
        DEMOS_POOL_SIZE = 10

        BATCH_SIZE = 8
