from pathlib import Path

from src.utils.MMLUConstants import MMLUConstants


class Constants:
    class DatasetsConstants:
        SCIQ = "sciq"
        RACE_ALL = "race_all"
        AI2_ARC_ARC_EASY = "ai2_arc.arc_easy"
        HELLASWAG = "hellaswag"
        DATASET_NAMES = [SCIQ, RACE_ALL, AI2_ARC_ARC_EASY]
        DATASET_NAMES.extend(MMLUConstants.MMLU_DATASETS_SAMPLE)

    class LLMProcessorConstants:
        MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
        LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
        LLAMA13B_MODEL = "meta-llama/Llama-2-13b-chat-hf"
        LLAMA70B_MODEL = "meta-llama/Llama-2-70b-chat-hf"
        GEMMA_MODEL = "google/gemma-7b-it"
        PHI_MODEL = "microsoft/phi-2"
        OLMO_MODEL = "allenai/OLMo-7B-Instruct"
        QWEN_MODEL = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8"
        QWEN2_MODEL = "Qwen/Qwen-7B-Chat-Int8"
        QWEN3_MODEL = "Qwen/Qwen1.5-MoE-A2.7B"

        PYTHIA_1B_MODEL = "EleutherAI/pythia-1b"
        PYTHIA_1B_MODEL_DEDUPED = "EleutherAI/pythia-1b-deduped"
        PYTHIA_2_8B_MODEL = "EleutherAI/pythia-2.8b"
        PYTHIA_2_8B_MODEL_DEDUPED = "EleutherAI/pythia-2.8b-deduped"
        PYTHIA_6_9B_MODEL = "EleutherAI/pythia-6.9b"
        PYTHIA_6_9B_MODEL_DEDUPED = "EleutherAI/pythia-6.9b-deduped"

        PYTHIA_MODELS = {"PYTHIA_1B": PYTHIA_1B_MODEL, "PYTHIA_1B_DEDUPED": PYTHIA_1B_MODEL_DEDUPED,
                            "PYTHIA_2_8B": PYTHIA_2_8B_MODEL, "PYTHIA_2_8B_DEDUPED": PYTHIA_2_8B_MODEL_DEDUPED,
                            "PYTHIA_6_9B": PYTHIA_6_9B_MODEL, "PYTHIA_6_9B_DEDUPED": PYTHIA_6_9B_MODEL_DEDUPED}

        MODEL_NAMES = {"MISTRAL": MISTRAL_MODEL, "LLAMA": LLAMA_MODEL,
                        "LLAMA13B": LLAMA13B_MODEL,
                        "LLAMA70B": LLAMA70B_MODEL,
                       "GEMMA": GEMMA_MODEL,
                       "OLMO": OLMO_MODEL}
        MODEL_NAMES.update(PYTHIA_MODELS)
        OLD_MODEL = {"QWEN": QWEN_MODEL, "QWEN2": QWEN2_MODEL, "QWEN3": QWEN3_MODEL}

        LOAD_IN_4BIT = False
        LOAD_IN_8BIT = True
        TRUST_REMOTE_CODE = False
        RETURN_TOKEN_TYPE_IDS = None
        BATCH_SIZE = 10

    class TemplatesGeneratorConstants:
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_INSTRUCTIONS_FOLDER_NAME = "MultipleChoiceTemplatesInstructions"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME
        DATA_PATH = Path(__file__).parents[2] / "Data"
        TEMPLATES_METADATA = "templates_metadata.csv"
        MMLU_DATASET_SIZES_PATH = DATA_PATH / "mmlu_datasets_sizes.csv"

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

        MAX_INSTANCES = 1600
        EVALUATE_ON_INFERENCE = ['test']
        EVALUATE_ON_ANALYZE = ['test']
        TEMPLATE_NUM = 0
        NUM_DEMOS = 0
        DEMOS_POOL_SIZE = 10

        BATCH_SIZE = 2

    class McNemarTestConstants:
        ALPHA = 0.05

    class ResultConstants:
        COMPARISON_MATRIX = "comparison_matrix"
        PERFORMANCE_SUMMARY = "performance_summary"
        BEST_COMBINATIONS = "best_combinations"
        CLUSTERING_RESULTS = "clustering_results"
        GROUPED_LEADERBOARD = "grouped_leaderboard"

        ACCURACY_COLUMN = "accuracy"
        CHOOSE_ACROSS_AXES = True
        NOT_CHOOSE_ACROSS_AXES = False

        ZERO_SHOT = "zero_shot"
        EMPTY_SYSTEM_FORMAT = "empty_system_format"

        GROUP = "group"

    class ClusteringConstants:
        RANDOM_STATE = 0

        K_MIN_INDEX = 2
        K_MAX_INDEX = 10

        MIN_CLUSTER_SIZE = 3
        MIN_SAMPLES = None

        CLUSTERING_METHODS = ["kmeans", "hdbscan", "spectral"]

    class BestCombinationsConstants:
        DATASET = "dataset"
        MODEL = "model"
        TOP_N = 5
        TEMPLATE = "template"