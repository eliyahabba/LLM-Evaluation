# selection_methods.py
import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from constants import ExperimentConfig


class SelectionMethod(ABC):
    """Base class for configuration selection methods."""

    def __init__(self, experiment_config: ExperimentConfig, full_rankings: pd.DataFrame = None):
        self.experiment_config = experiment_config
        self.full_rankings = full_rankings

    @abstractmethod
    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """Select optimal configuration based on sample data."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Return a user-friendly display name for the method."""
        pass


class RandomSelection(SelectionMethod):
    """Implements random selection method."""

    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """Randomly select one configuration from all possible configurations."""
        # Get unique configurations
        unique_configs = (
            data[self.experiment_config.get_config_axes()]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Randomly select one configuration
        random_idx = np.random.randint(len(unique_configs))
        selected_config = unique_configs.iloc[random_idx]

        return selected_config.to_dict()

    def get_display_name(self) -> str:
        return "Random Baseline"


class RegressionBasedSelection(SelectionMethod):
    """Implements regression-based selection method."""

    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """
        Select configuration using linear regression predictions.

        Process:
        1. Prepare training data from the sample:
           - Group by configuration to get mean scores
           - Create feature matrix using one-hot encoding

        2. Train regression model:
           - Use configurations from sample as features
           - Use mean scores from sample as target

        3. Generate predictions:
           - Use ALL configurations from full rankings table
           - Predict scores for all possible configurations

        4. Select best configuration:
           - Choose configuration with highest predicted score
        """
        logger = logging.getLogger(__name__)
        config_axes = self.experiment_config.get_config_axes()

        # Step 1: Prepare training data from sample
        logger.info("Preparing training data from sample")
        grouped_data = (
            data
            .groupby(config_axes)['score']
            .agg(['mean', 'count'])
            .reset_index()
        )

        # Create feature matrix using one-hot encoding
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train = encoder.fit_transform(grouped_data[config_axes])
        y_train = grouped_data['mean'].values

        # Step 2: Train regression model
        logger.info("Training regression model")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 3: Generate predictions for ALL possible configurations
        logger.info("Generating predictions for all configurations")

        # Get all unique configurations from the full rankings
        all_configs = (
            self.full_rankings[config_axes]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Transform all configurations to feature matrix
        X_all = encoder.transform(all_configs[config_axes])

        # Make predictions
        predictions = model.predict(X_all)

        # Step 4: Select best configuration
        best_idx = np.argmax(predictions)
        selected_config = all_configs.iloc[best_idx]

        logger.info("Selected configuration based on regression predictions")
        return selected_config.to_dict()

    def get_display_name(self) -> str:
        return "Regression Predictor"


class AxiswiseSelection(SelectionMethod):
    """Implements axis-wise selection method."""

    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """Find optimal value for each axis independently."""
        config_axes = self.experiment_config.get_config_axes()
        optimal_config = {}

        # For each configuration axis
        for col in config_axes:
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

    def get_display_name(self) -> str:
        return "Axis-wise Selection"


class MajoritySelection(SelectionMethod):
    """Implements majority selection method."""

    def select_configuration(self, data: pd.DataFrame) -> Dict:
        """Find configuration with best performance on the sample."""
        # Group by configuration and get mean scores
        config_scores = (
            data
            .groupby(self.experiment_config.config_axes)['score']
            .mean()
            .reset_index()
        )

        # Get best configuration
        best_config = config_scores.loc[
            config_scores['score'].idxmax(),
            self.experiment_config.config_axes
        ]

        return best_config.to_dict()

    def get_display_name(self) -> str:
        return "Best Global Configuration"
