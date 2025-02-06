# data_processor.py
import pandas as pd
from typing import List, Optional
from constants import DATA_CONFIG


class DataProcessor:
    """Handles data loading, cleaning, and optimization operations."""

    def __init__(self, input_file: str):
        """Initialize with path to input file."""
        self.input_file = input_file
        self.data = None

    def load_and_clean_data(self) -> pd.DataFrame:
        """Load data and drop unnecessary columns, keeping only required ones."""
        self.data = pd.read_parquet(
            self.input_file,
            columns=DATA_CONFIG['REQUIRED_COLUMNS']
        )
        return self.data

    def get_configuration_axes(self) -> List[str]:
        """Return list of configuration parameter columns."""
        return DATA_CONFIG['CONFIG_AXES']
