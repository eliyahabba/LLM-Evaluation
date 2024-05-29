from enum import Enum
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
        MISTRAL_V1_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
        MISTRAL_V2_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
        MISTRAL_V3_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
        MISTRAL_FAMILY = [MISTRAL_V2_MODEL, MISTRAL_V3_MODEL]

        LLAMA7B_MODEL = "meta-llama/Llama-2-7b-chat-hf"
        LLAMA13B_MODEL = "meta-llama/Llama-2-13b-chat-hf"
        LLAMA70B_MODEL = "meta-llama/Llama-2-70b-chat-hf"
        LLAMA3_8B_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
        LLAMA7B_BASE_MODEL = "meta-llama/Llama-2-7b-hf"
        LLAMA13B_BASE_MODEL = "meta-llama/Llama-2-13b-hf"

        LLAMAS_FAMILY = [LLAMA7B_MODEL, LLAMA13B_MODEL, LLAMA3_8B_MODEL]

        GEMMA_7B_MODEL = "google/gemma-7b-it"
        GEMMA_2B_MODEL = "google/gemma-2b-it"
        GEMMA_FAMILY = [GEMMA_7B_MODEL, GEMMA_2B_MODEL]

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

        PHI3_MEDIUM_MODEL = "microsoft/Phi-3-medium-4k-instruct"
        PHI3_SMALL_MODEL = "microsoft/Phi-3-small-8k-instruct"
        PHI3_MINI_MODEL = "microsoft/Phi-3-mini-4k-instruct"
        PHI_FAMILY = [PHI3_MEDIUM_MODEL, PHI3_SMALL_MODEL, PHI3_MINI_MODEL]
        PHI_FAMILY = [PHI3_MEDIUM_MODEL, PHI3_MINI_MODEL]

        PYTHIA_MODELS = {"PYTHIA_1B": PYTHIA_1B_MODEL, "PYTHIA_1B_DEDUPED": PYTHIA_1B_MODEL_DEDUPED,
                         "PYTHIA_2_8B": PYTHIA_2_8B_MODEL, "PYTHIA_2_8B_DEDUPED": PYTHIA_2_8B_MODEL_DEDUPED,
                         "PYTHIA_6_9B": PYTHIA_6_9B_MODEL, "PYTHIA_6_9B_DEDUPED": PYTHIA_6_9B_MODEL_DEDUPED}

        BASE_MODEL_NAMES = {"PHI_MINI": PHI3_MINI_MODEL,
                            "PHI_SMALL": PHI3_SMALL_MODEL,
                            "PHI_MEDIUM": PHI3_MEDIUM_MODEL,
                            "MISTRAL_V1": MISTRAL_V1_MODEL,
                            "MISTRAL_V2": MISTRAL_V2_MODEL,
                            "MISTRAL_V3": MISTRAL_V3_MODEL,
                            "LLAMA7B": LLAMA7B_MODEL,
                            "LLAMA13B": LLAMA13B_MODEL,
                            "LLAMA70B": LLAMA70B_MODEL,
                            "LLAMA3_8B": LLAMA3_8B_MODEL,
                            "LLAMA7B_BASE": LLAMA7B_BASE_MODEL,
                            "LLAMA13B_BASE": LLAMA13B_BASE_MODEL,
                            "GEMMA_7B": GEMMA_7B_MODEL,
                            "GEMMA_2B": GEMMA_2B_MODEL,
                            "OLMO": OLMO_MODEL}
        MODEL_NAMES = BASE_MODEL_NAMES
        MODEL_NAMES.update(PYTHIA_MODELS)

        MODELS_FAMILIES = {"Mistral": MISTRAL_FAMILY, "Llama": LLAMAS_FAMILY, "gemma": GEMMA_FAMILY, "Phi3": PHI_FAMILY}

        OLD_MODEL = {"QWEN": QWEN_MODEL, "QWEN2": QWEN2_MODEL, "QWEN3": QWEN3_MODEL}

        LOAD_IN_4BIT = False
        LOAD_IN_8BIT = True
        TRUST_REMOTE_CODE = False
        RETURN_TOKEN_TYPE_IDS = None

    class TemplatesGeneratorConstants:
        MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME = "MultipleChoiceTemplatesStructured"
        MULTIPLE_CHOICE_INSTRUCTIONS_FOLDER_NAME = "MultipleChoiceTemplatesInstructions"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME
        DATA_PATH = Path(__file__).parents[2] / "Data"
        TEMPLATES_METADATA = "templates_metadata.csv"
        MMLU_DATASET_SIZES_PATH = DATA_PATH / "mmlu_datasets_sizes.csv"

    class ExperimentConstants:
        TEMPLATES_RANGE = [0, 1]
        MAIN_RESULTS_FOLDER_NAME = "results"
        MAIN_RESULTS_PATH = Path(__file__).parents[2] / MAIN_RESULTS_FOLDER_NAME

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

        BATCH_SIZE = 8

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
        TEMPLATE_NAME = "template_name"

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

    class BestOrWorst(Enum):
        BEST = "best"
        WORST = "worst"
