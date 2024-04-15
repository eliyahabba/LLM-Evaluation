from pathlib import Path


class Constants:
    class DatasetsConstants:
        SCIQ = "sciq"
        RACE_ALL = "race_all"
        AI2_ARC_ARC_EASY = "ai2_arc.arc_easy"
        MMLU_GENERAL = "mmlu"
        HELLASWAG = "hellaswag"
        MMLU_DATASETS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
                         'college_biology',
                         'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine',
                         'college_physics',
                         'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering',
                         'elementary_mathematics',
                         'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
                         'high_school_computer_science',
                         'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
                         'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
                         'high_school_physics',
                         'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
                         'high_school_world_history',
                         'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
                         'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
                         'moral_disputes',
                         'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
                         'professional_law',
                         'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
                         'sociology',
                         'us_foreign_policy', 'virology', 'world_religions']
        MMLU_DATASETS_SAMPLE = ['mmlu.anatomy', 'mmlu.college_computer_science',
                                'mmlu.electrical_engineering', 'mmlu.elementary_mathematics',
                                'mmlu.global_facts', 'mmlu.machine_learning',
                                'mmlu.medical_genetics', 'mmlu.professional_accounting']
        DATASET_NAMES = [SCIQ, RACE_ALL, AI2_ARC_ARC_EASY]
        DATASET_NAMES.extend(MMLU_DATASETS_SAMPLE)

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

        MAX_INSTANCES = 100
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

    class ClusteringConstants:
        RANDOM_STATE = 0

        K_MIN_INDEX = 2
        K_MAX_INDEX = 10

        MIN_CLUSTER_SIZE = 3
        MIN_SAMPLES = None

        CLUSTERING_METHODS = ["kmeans", "hdbscan"]