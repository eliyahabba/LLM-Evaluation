# constants.py
from typing import List, Dict

# Model configurations
MODELS = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'allenai/OLMoE-1B-7B-0924-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]
MODELS = [
    'allenai/OLMoE-1B-7B-0924-Instruct',
]

# Data configuration
DATA_CONFIG = {
    'REQUIRED_COLUMNS': [
        'dataset', 'sample_index', 'model',
        'shots', 'template', 'separator', 'enumerator', 'choices_order',
        'score'
    ],
    'CONFIG_AXES': [
        'shots', 'template', 'separator', 'enumerator', 'choices_order'
    ]
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    'SAMPLE_SIZES': [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000],
    'N_ITERATIONS': 3,
    'RANDOM_SEED': 42
}

# File paths configuration
def get_file_paths(model_name: str) -> Dict[str, str]:
    """Generate standardized file paths for a given model."""
    model_id = model_name.split('/')[-1]
    return {
        'input': f'data/results_{model_id}.parquet',
        'rankings': f'output/rankings/{model_id}.parquet',
        'results': f'output/results/{model_id}.json',
        'plot': f'output/plots/{model_id}.png'
    }
