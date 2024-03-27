from pathlib import Path


class Constants:
    class LLMProcessorConstants:
        MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
        LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
        GEMMA_MODEL = "google/gemma-7b-it"
        PHI_MODEL = "microsoft/phi-2"
        QWEN_MODEL = "Qwen/Qwen1.5-7B-Chat"
        MODEL_NAMES = {"PHI": PHI_MODEL, "MISTRAL": MISTRAL_MODEL, "LLAMA": LLAMA_MODEL, "GEMMA": GEMMA_MODEL,
                       "QWEN": QWEN_MODEL}

        LOAD_IN_4BIT = False
        LOAD_IN_8BIT = True
        TRUST_REMOTE_CODE = False

    class TemplatesGeneratorConstants:
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME

    class ExperimentConstants:
        TEMPLATES_RANGE = [0, 1]
        MAIN_RESULTS_FOLDER_NAME = "results"
        MAIN_RESULTS_PATH = Path(__file__).parents[2] / MAIN_RESULTS_FOLDER_NAME

        STRUCTURED_INPUT_FOLDER = "structured_input"
        NOT_STRUCTURED_INPUT_FOLDER = "not_structured_input"
        STRUCTURED_INPUT_FOLDER_PATH = MAIN_RESULTS_PATH / STRUCTURED_INPUT_FOLDER
        NOT_STRUCTURED_INPUT_FOLDER_PATH = MAIN_RESULTS_PATH / NOT_STRUCTURED_INPUT_FOLDER

        EMPTY_SYSTEM_FORMATS = "formats.empty"
        LLAMA_SYSTEM_FORMATS = "formats.llama"
        SYSTEM_FORMATS = [EMPTY_SYSTEM_FORMATS, LLAMA_SYSTEM_FORMATS]
        SYSTEM_FORMATS_NAMES = {EMPTY_SYSTEM_FORMATS: "empty_system_format",
                                LLAMA_SYSTEM_FORMATS: "llama_system_format"}
        SYSTEM_FORMAT_INDEX = 0

        MAX_INSTANCES = 100
        EVALUATE_ON = ['test']
        TEMPLATE_NUM = 0
        NUM_DEMOS = 0
        DEMOS_POOL_SIZE = 10

        BATCH_SIZE = 10
