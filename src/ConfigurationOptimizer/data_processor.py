# data_processor.py
from typing import List

import pandas as pd

from constants import ExperimentConfig


class DataProcessor:
    """Handles data loading, cleaning, and optimization operations."""

    def __init__(self, input_file: str, experiment_config: ExperimentConfig):
        """Initialize with path to input file and experiment configuration."""
        self.input_file = input_file
        self.experiment_config = experiment_config
        self.data = None

    def load_and_clean_data(self) -> pd.DataFrame:
        """Load data and drop unnecessary columns, keeping only required ones."""
        self.data = pd.read_parquet(
            self.input_file,
            columns=self.experiment_config.required_columns
        )

        return self.data

    def get_configuration_axes(self) -> List[str]:
        """Return list of configuration parameter columns."""
        return self.experiment_config.get_config_axes()
