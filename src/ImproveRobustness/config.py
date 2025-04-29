"""
Configuration module for the prompt dimension robustness experiment.
Contains all configurable parameters for the experiment.
"""

from pathlib import Path
import random
import numpy as np
import torch


class ExperimentConfig:
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent
    
    # Data paths
    RAW_DATA_DIR = Path("/Users/ehabba/PycharmProjects/LLM-Evaluation/src/ConfigurationOptimizer/data")  # Source of raw data files
    DATA_DIR = BASE_DIR / "data"  # Directory for processed data files
    
    # Output paths
    OUTPUT_DIR = BASE_DIR / "results"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    MODELS_DIR = OUTPUT_DIR / "models"

    # Experiment setup
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    TARGET_DIMENSION = "dimensions_1: enumerator"

    # All possible values for the target dimension
    ALL_DIMENSION_VALUES = ['keyboard', 'lowercase', 'numbers', 'greek', 'roman', 'capitals']

    # Values to use in training (4 out of 6)
    TRAINING_DIMENSION_VALUES = ["keyboard", "capitals", "roman"]

    # Values to exclude from training (2 out of 6)
    EXCLUDED_DIMENSION_VALUES = ["lowercase", "numbers", "greek"]

    # Datasets to use for training
    TRAINING_DATASETS = [
        "mmlu_pro.engineering",
        "mmlu.medical_genetics",
        "hellaswag",
        "mmlu.college_medicine",
        "mmlu.global_facts",
        "mmlu.high_school_biology",
    ]

    # Dataset to use for testing
    TEST_DATASETS = [
        "mmlu.anatomy",
        "ai2_arc.arc_easy",
        "mmlu.human_aging"
    ]

    # Training parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4
    FP16 = True
    SEED = 42
    
    @staticmethod
    def set_seed():
        """Set random seeds for reproducibility."""
        random.seed(ExperimentConfig.SEED)
        np.random.seed(ExperimentConfig.SEED)
        torch.manual_seed(ExperimentConfig.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ExperimentConfig.SEED) 