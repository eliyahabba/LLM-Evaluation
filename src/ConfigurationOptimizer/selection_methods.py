# selection_methods.py
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict
from constants import DATA_CONFIG


class SelectionMethod(ABC):
    """Base class for configuration selection methods."""

    @abstractmethod
    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """Select optimal configuration based on sample data."""
        pass


class MajoritySelection(SelectionMethod):
    """Implements majority selection method."""

    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """Find configuration with best performance on the sample."""
        # Group by configuration and get mean scores
        config_scores = (
            data
            .groupby(DATA_CONFIG['CONFIG_AXES'])['score']
            .mean()
            .reset_index()
        )

        # Get best configuration
        best_config = config_scores.loc[
            config_scores['score'].idxmax(),
            DATA_CONFIG['CONFIG_AXES']
        ]

        return best_config.to_dict()


class AxiswiseSelection(SelectionMethod):
    """Implements axis-wise selection method."""

    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """Find optimal value for each axis independently."""
        optimal_config = {}

        # For each configuration axis
        for col in DATA_CONFIG['CONFIG_AXES']:
            # Group by this axis and get mean scores
            axis_scores = (
                data
                .groupby(col)['score']
                .mean()
                .reset_index()
            )

            # Get best value for this axis
            optimal_config[col] = axis_scores.loc[
                axis_scores['score'].idxmax(),
                col
            ]

        return optimal_config