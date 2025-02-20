from pathlib import Path


class ProcessingConstants:
    """Constants for data processing configuration."""
    
    # Processing batch sizes and limits
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_NUM_WORKERS = 1
    DEFAULT_NUM_BATCHES = 10
    
    # Directory structure
    TEMP_DIR_NAME = "temp"
    FULL_SCHEMA_DIR_NAME = "full_schema"
    LEAN_SCHEMA_DIR_NAME = "lean_schema"
    PROCESSED_FILES_RECORD = "processed_files.txt"
    
    # File paths
    INPUT_DATA_DIR = "/Users/ehabba/Desktop/IBM_Results"
    INPUT_DATA_DIR = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full"  # For HF downloads
    OUTPUT_DATA_DIR = "/Users/ehabba/Desktop/IBM_Results_Processed"
    OUTPUT_DATA_DIR = "/cs/snapless/gabis/eliyahabba/processed_data"  # For processed files

    MODELS_METADATA_PATH = Path(__file__).parents[2] / "src" / "utils" / "models_metadata.json"
    
    # File extensions
    PARQUET_EXTENSION = ".parquet"
    LOCK_FILE_EXTENSION = ".lock"
    
    # HuggingFace repositories
    SOURCE_REPO = "OfirArviv/HujiCollabOutput"
    OUTPUT_REPO = "eliyahabba/HujiCollabOutput"


class DatasetRepos:
    """Repository mappings for different datasets."""
    
    REPOS = {
        "mmlu": "cais/mmlu",
        "mmlu_pro": "TIGER-Lab/MMLU-Pro",
        "hellaswag": "Rowan/hellaswag",
        "openbookqa": "allenai/openbookqa",
        "social_i_qa": "allenai/social_i_qa",
        "ai2_arc_challenge": "allenai/ai2_arc/ARC-Challenge",
        "ai2_arc_easy": "allenai/ai2_arc/ARC-Easy",
        "quaily": "emozilla/quality",
        "global_mmlu_lite": "CohereForAI/Global-MMLU-Lite",
        "global_mmlu": "CohereForAI/Global-MMLU"
    }


class SchemaConstants:
    """Constants for schema configuration."""
    
    # Default schema values
    DEFAULT_LANGUAGE = "en"
    DEFAULT_TASK_TYPE = "classification"
    DEFAULT_PROMPT_CLASS = "MultipleChoice"
    DEFAULT_EVALUATION_METHOD = "content_similarity"
    DEFAULT_EVALUATION_DESCRIPTION = "Finds the most similar answer among the given choices by comparing the textual content"
    
    # Lean schema columns
    LEAN_SCHEMA_COLUMNS = [
        "evaluation_id",
        "dataset",
        "sample_index",
        "model",
        "quantization",
        "shots",
        "instruction_phrasing_text",
        "instruction_phrasing_name",
        "separator",
        "enumerator",
        "choices_order",
        "raw_input",
        "generated_text",
        "cumulative_logprob",
        "closest_answer",
        "ground_truth",
        "score"
    ]


class ParquetConstants:
    """Constants for Parquet file operations."""
    
    VERSION = "2.6"
    WRITE_STATISTICS = True 