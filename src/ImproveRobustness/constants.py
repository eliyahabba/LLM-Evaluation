"""
Constants module for the dimension robustness experiment.
Contains fixed values that are not meant to be changed between runs.
"""

# Model and tokenizer configuration constants
LORA_CONFIG = {
    "R": 8,
    "LORA_ALPHA": 16,
    "LORA_DROPOUT": 0.1,
    "BIAS": "none",
    "TASK_TYPE": "CAUSAL_LM",
}

# Training constants
TRAINING_ARGS = {
    "OPTIM": "paged_adamw_32bit",
    "FP16": True,
    "BF16": False,
    "MAX_STEPS": -1,
    "GROUP_BY_LENGTH": True,
    "EVALUATION_STRATEGY": "steps",
}

# Dimension values mapping
DIMENSION_VALUES = {
    "ENUMERATOR": {
        "ALL": ['keyboard', 'lowercase', 'numbers', 'greek', 'roman', 'capitals'],
        "DESCRIPTION": "Character set used for enumerating choices (A, B, C vs 1, 2, 3, etc.)"
    },
    "SEPARATOR": {
        "ALL": [';', '.', ')', ':', '-'],
        "DESCRIPTION": "Character used to separate the enumerator from the choice"
    },
    "CHOICES_ORDER": {
        "ALL": ["original", "shortest_to_longest", "longest_to_shortest", "alphabetical"],
        "DESCRIPTION": "Order in which choices are presented"
    }
}

# Available datasets
ALL_DATASETS = [
    # STEM
    "mmlu.medical_genetics",
    "mmlu.college_medicine",
    "mmlu.high_school_biology",
    "mmlu.anatomy",
    # HUMANITIES
    "mmlu.global_facts",
    "hellaswag",
    "mmlu.human_aging",
    # REASONING
    "mmlu_pro.engineering",
    "ai2_arc.arc_easy"
]

# Generation constants
GENERATION_CONFIG = {
    "MAX_NEW_TOKENS": 50,
    "DO_SAMPLE": False,
    "NUM_BEAMS": 1,
    "EARLY_STOPPING": True
}
