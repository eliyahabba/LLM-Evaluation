from pathlib import Path


class Constants:
    class LLMProcessorConstants:
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
        LOAD_IN_4BIT = False
        LOAD_IN_8BIT = True

    class TemplatesGeneratorConstants:
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME

    class ExperimentConstants:
        TEMPLATES_RANGE = [0, 1]
        RESULTS_WITHOUT_STRUCTURE_FOLDER_NAME = "results_without_structure"
        RESULTS_FOLDER_NAME = "results"
        RESULTS_WITHOUT_STRUCTURE_PATH = Path(__file__).parents[2] / RESULTS_WITHOUT_STRUCTURE_FOLDER_NAME
        RESULTS_PATH = Path(__file__).parents[2] / RESULTS_FOLDER_NAME
        RESULTS_PATHS = [RESULTS_WITHOUT_STRUCTURE_PATH, RESULTS_PATH]

        EMPTY_SYSTEM_FORMATS = "formats.empty"
        MISTRAL_SYSTEM_FORMATS = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
        SYSTEM_FORMATS = {"empty_system_format": EMPTY_SYSTEM_FORMATS, "mistral_system_format": MISTRAL_SYSTEM_FORMATS}
        SYSTEM_FORMAT_INDEX = 0

        MAX_INSTANCES = 100
        EVALUATE_ON = ['test']
        TEMPLATE_NUM = 0
        NUM_DEMOS = 0
        DEMOS_POOL_SIZE = 10

        BATCH_SIZE = 10
