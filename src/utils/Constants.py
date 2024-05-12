from pathlib import Path

from src.utils.MMLUConstants import MMLUConstants


class Constants:
    class DatasetsConstants:
        SCIQ = "sciq"
        RACE_ALL = "race_all"
        AI2_ARC_ARC_EASY = "ai2_arc.arc_easy"
        MMLU_GENERAL = "mmlu"
        HELLASWAG = "hellaswag"
        # MMLU_DATASETS_SAMPLE = ['mmlu.abstract_algebra',
        #                         'mmlu.anatomy',
        #                         'mmlu.astronomy',
        #                         'mmlu.business_ethics'
        #                         'mmlu.clinical_knowledge',
        #
        #                         'mmlu.college_biology',
        #                         'mmlu.college_computer_science',
        #
        #                         'mmlu.college_chemistry',
        #                         'mmlu.college_mathematics', 'mmlu.college_medicine',
        #                         'mmlu.college_physics',
        #                         'mmlu.computer_security', 'mmlu.conceptual_physics', 'mmlu.econometrics',
        #                         'mmlu.formal_logic', 'mmlu.high_school_biology', 'mmlu.high_school_chemistry',
        #                         'mmlu.high_school_computer_science',
        #                         'mmlu.high_school_european_history',
        #
        #                         'mmlu.high_school_geography', 'mmlu.high_school_government_and_politics',
        #                         'mmlu.high_school_macroeconomics', 'mmlu.high_school_mathematics', 'mmlu.high_school_microeconomics',
        #                         'mmlu.high_school_physics',
        #
        #                         'mmlu.high_school_psychology', 'mmlu.high_school_statistics', 'mmlu.high_school_us_history',
        #                         'mmlu.high_school_world_history',
        #                         'mmlu.human_aging', 'mmlu.human_sexuality', 'mmlu.international_law', 'mmlu.jurisprudence', 'mmlu.logical_fallacies',
        #
        #
        #                         'mmlu.machine_learning', 'mmlu.management', 'mmlu.marketing', 'mmlu.medical_genetics', 'mmlu.miscellaneous',
        #                         'mmlu.moral_disputes',
        #                         'mmlu.moral_scenarios', 'mmlu.nutrition', 'mmlu.philosophy', 'mmlu.prehistory', 'mmlu.professional_accounting',
        #
        #                         'mmlu.professional_law',
        #                         'mmlu.professional_medicine', 'mmlu.professional_psychology', 'mmlu.public_relations', 'mmlu.security_studies',
        #                         'mmlu.sociology',
        #                         'mmlu.us_foreign_policy', 'mmlu.virology', 'mmlu.world_religions',
        #
        #
        #                         'mmlu.electrical_engineering', 'mmlu.elementary_mathematics',
        #                         'mmlu.global_facts', 'mmlu.machine_learning',
        #                         'mmlu.medical_genetics', 'mmlu.professional_accounting']
        DATASET_NAMES = [SCIQ, RACE_ALL, AI2_ARC_ARC_EASY]
        DATASET_NAMES.extend(MMLUConstants.MMLU_DATASETS_SAMPLE)

    class LLMProcessorConstants:
        MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
        LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
        GEMMA_MODEL = "google/gemma-7b-it"
        PHI_MODEL = "microsoft/phi-2"
        OLMO_MODEL = "allenai/OLMo-7B-Instruct"
        QWEN_MODEL = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8"
        QWEN2_MODEL = "Qwen/Qwen-7B-Chat-Int8"
        QWEN3_MODEL = "Qwen/Qwen1.5-MoE-A2.7B"
        MODEL_NAMES = {"PHI": PHI_MODEL, "MISTRAL": MISTRAL_MODEL, "LLAMA": LLAMA_MODEL, "GEMMA": GEMMA_MODEL,
                       "OLMO": OLMO_MODEL}
        OLD_MODEL = {"QWEN": QWEN_MODEL, "QWEN2": QWEN2_MODEL, "QWEN3": QWEN3_MODEL}

        LOAD_IN_4BIT = False
        LOAD_IN_8BIT = True
        TRUST_REMOTE_CODE = False
        RETURN_TOKEN_TYPE_IDS = None
        BATCH_SIZE = 10

    class TemplatesGeneratorConstants:
        MULTIPLE_CHOICE_FOLDER_NAME = "MultipleChoiceTemplates"
        MULTIPLE_CHOICE_PATH = Path(__file__).parents[2] / "Data" / MULTIPLE_CHOICE_FOLDER_NAME

        TEMPLATES_METADATA = "templates_metadata.csv"

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

        MAX_INSTANCES = 1000
        EVALUATE_ON = ['test']
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