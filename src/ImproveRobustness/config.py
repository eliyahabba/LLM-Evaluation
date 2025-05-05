"""
Configuration module for the prompt dimension robustness experiment.
Contains all configurable parameters for the experiment.
"""

from pathlib import Path
import random
import os
import numpy as np
import torch
from dotenv import load_dotenv

from src.ImproveRobustness.constants import (
    LORA_CONFIG, 
    TRAINING_ARGS, 
    DIMENSION_VALUES, 
    ALL_DATASETS,
    GENERATION_CONFIG,
)

# Load environment variables from .env file
load_dotenv()


class ExperimentConfig:
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"  # Directory for processed data files
    RAW_DATA_DIR = DATA_DIR  / "raw_data"  # Directory for raw data files

    # Output paths
    RESULTS_DIR = BASE_DIR / "results"
    PLOTS_DIR = RESULTS_DIR / "plots"
    MODELS_DIR = RESULTS_DIR / "models"

    # Experiment setup
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    TARGET_DIMENSION = "dimensions_1: enumerator"

    # All possible values for the target dimension - imported from constants
    ALL_DIMENSION_VALUES = DIMENSION_VALUES["ENUMERATOR"]["ALL"]

    # Values to use in training (3 out of 6)
    TRAINING_DIMENSION_VALUES = ["keyboard", "capitals", "roman"]

    # Values to exclude from training (3 out of 6)
    EXCLUDED_DIMENSION_VALUES = ["lowercase", "numbers", "greek"]

    # Datasets to use for training - 6 out of 9 datasets
    TRAINING_DATASETS = [
        "mmlu.medical_genetics",
        "mmlu.college_medicine", 
        "mmlu.high_school_biology",
        "mmlu.global_facts",
        "hellaswag",
        "mmlu_pro.engineering"
    ]

    # Dataset to use for testing - 3 out of 9 datasets
    TEST_DATASETS = [
        "mmlu.anatomy",
        "ai2_arc.arc_easy",
        "mmlu.human_aging"
    ]

    # Training parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4
    FP16 = TRAINING_ARGS["FP16"]
    SEED = 42
    
    # Trainer class default parameters
    ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN", None)  # Get token from environment variable
    QUANT_CONFIG_TYPE = None  # Quantization config, options: None, "4bit", "8bit"
    USE_LORA = True  # Whether to use LoRA for fine-tuning
    
    # Training run configuration
    EVAL_BEFORE_FINETUNING = True  # Evaluate model before fine-tuning
    DO_FINETUNING = True  # Perform fine-tuning
    EVAL_AFTER_FINETUNING = True  # Evaluate model after fine-tuning
    
    # Training arguments defaults
    PER_DEVICE_TRAIN_BATCH_SIZE = 2
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    LOGGING_STEPS = 10
    SAVE_STEPS = 20
    SAVE_TOTAL_LIMIT = 3
    MAX_GRAD_NORM = 0.3
    LR_SCHEDULER_TYPE = "constant"
    
    # LoRA configuration - imported from constants
    LORA_R = LORA_CONFIG["R"]
    LORA_ALPHA = LORA_CONFIG["LORA_ALPHA"]
    LORA_DROPOUT = LORA_CONFIG["LORA_DROPOUT"]
    
    # Generation parameters - imported from constants
    MAX_NEW_TOKENS = GENERATION_CONFIG["MAX_NEW_TOKENS"]
    DO_SAMPLE = GENERATION_CONFIG["DO_SAMPLE"]
    
    @staticmethod
    def set_seed():
        """Set random seeds for reproducibility."""
        random.seed(ExperimentConfig.SEED)
        np.random.seed(ExperimentConfig.SEED)
        torch.manual_seed(ExperimentConfig.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ExperimentConfig.SEED)

    @staticmethod
    def print_config_info():
        """Print important configuration information."""
        print("\n=== Configuration Info ===")
        print(f"Model: {ExperimentConfig.MODEL_NAME}")
        print(f"Target dimension: {ExperimentConfig.TARGET_DIMENSION}")
        print(f"Training on values: {ExperimentConfig.TRAINING_DIMENSION_VALUES}")
        print(f"Excluding values: {ExperimentConfig.EXCLUDED_DIMENSION_VALUES}")
        
        # Print HF token status (but not the actual token)
        if ExperimentConfig.ACCESS_TOKEN:
            print("HF Access Token: Available")
        else:
            print("WARNING: HF Access Token not found in environment variables.")
            print("You may need to add HF_ACCESS_TOKEN to your .env file or environment.")
        print("===========================\n") 