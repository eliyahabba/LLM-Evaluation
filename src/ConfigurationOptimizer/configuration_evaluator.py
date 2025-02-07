# configuration_evaluator.py
import logging

import pandas as pd
import os
from typing import Dict, Optional

from constants import DATA_CONFIG


class ConfigurationEvaluator:
    """Handles computation and storage of configuration rankings."""

    def __init__(self, data: pd.DataFrame, output_file: str):
        self.data = data
        self.output_file = output_file
        self.rankings = None

    def compute_rankings(self, force_recompute: bool = False) -> pd.DataFrame:
        """Compute or load configuration rankings."""
        logger = logging.getLogger(__name__)

        if not force_recompute and os.path.exists(self.output_file):
            logger.info("Loading existing rankings file")
            self.rankings = pd.read_parquet(self.output_file)
            return self.rankings

        logger.info("Computing configuration rankings...")

        # Group by configuration axes and compute average score
        self.rankings = (
            self.data
            .groupby(DATA_CONFIG['CONFIG_AXES'])['score']
            .agg(['mean', 'count'])
            .reset_index()
            .sort_values('mean', ascending=False)
        )

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Save rankings
        self.rankings.to_parquet(self.output_file)
        return self.rankings
