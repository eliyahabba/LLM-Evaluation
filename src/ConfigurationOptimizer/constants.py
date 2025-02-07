# Data configuration
from typing import Dict, List, Any

import pandas as pd


class ExperimentConfig:
    """Configuration class for experiment settings and filters."""

    def __init__(
            self,
            required_columns: List[str] = None,
            config_axes: List[str] = None,
            filters: Dict[str, Any] = None,
            experiment_suffix: str = ""
    ):
        self.required_columns = required_columns or [
            'dataset', 'sample_index', 'model',
            'shots', 'template', 'separator', 'enumerator', 'choices_order',
            'score'
        ]
        self.config_axes = config_axes or [
            'shots', 'template', 'separator', 'enumerator', 'choices_order'
        ]
        self.filters = filters or {}
        self.experiment_suffix = experiment_suffix

    def apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply configured filters to the dataset."""
        filtered_data = data
        for column, value in self.filters.items():
            filtered_data = filtered_data[filtered_data[column] == value]
        return filtered_data

    def get_config_axes(self) -> List[str]:
        """Get configuration axes excluding filtered columns."""
        return [axis for axis in self.config_axes if axis not in self.filters]


# Example configurations
DEFAULT_CONFIG = ExperimentConfig()

ZERO_SHOT_CONFIG = ExperimentConfig(
    filters={'shots': 0},
    experiment_suffix="_zero_shot"
)

# Experiment configuration
EXPERIMENT_ARGS_CONFIG = {
    'SAMPLE_SIZES': [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000],
    'N_ITERATIONS': 3,
    'RANDOM_SEED': 42
}

# Model configurations
MODELS = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'allenai/OLMoE-1B-7B-0924-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]



# File paths configuration
def get_file_paths(model_name: str, experiment_config: ExperimentConfig) -> Dict[str, str]:
    """Generate standardized file paths for a given model and experiment configuration."""
    model_id = model_name.split('/')[-1]
    suffix = experiment_config.experiment_suffix
    return {
        'input': f'data/results_{model_id}.parquet',
        'rankings': f'output/rankings/{model_id}.parquet',
        'results': f'output/results/{model_id}.json',
        'plot': f'output/plots/{model_id}.png'
    }
