# experiment_runner.py
import json
import logging
import os
from typing import List, Dict

import numpy as np
import pandas as pd

from constants import EXPERIMENT_ARGS_CONFIG
from selection_methods import SelectionMethod


class ExperimentRunner:
    """Manages experiment iterations and evaluations."""

    def __init__(
            self,
            data: pd.DataFrame,
            rankings: pd.DataFrame,
            selection_methods: List[SelectionMethod],
            results_file: str
    ):
        self.data = data
        self.rankings = rankings
        self.selection_methods = selection_methods
        self.results_file = results_file
        self.results = {}

    def run_experiments(
            self,
            sample_sizes: List[int] = EXPERIMENT_ARGS_CONFIG['SAMPLE_SIZES'],
            n_iterations: int = EXPERIMENT_ARGS_CONFIG['N_ITERATIONS'],
            base_seed: int = EXPERIMENT_ARGS_CONFIG['RANDOM_SEED']
    ) -> Dict:
        """Run experiments for different sample sizes and methods."""
        logger = logging.getLogger(__name__)

        # Initialize results dictionary for all methods
        for method in self.selection_methods:
            method_name = method.get_display_name()  # Use new display name method
            self.results[method_name] = {
                'sample_sizes': sample_sizes,
                'gaps': [],
                'gap_stds': [],  # Add standard deviation tracking
                'raw_gaps': []  # Store all gaps for each sample size
            }

        # Generate different seeds for each iteration
        iteration_seeds = [base_seed + i for i in range(n_iterations)]

        # For each sample size
        for size in sample_sizes:
            logger.info(f"Processing sample size: {size}")

            # Store gaps for each method at this sample size
            method_gaps = {method.get_display_name(): [] for method in self.selection_methods}

            # For each iteration
            for iteration, seed in enumerate(iteration_seeds):
                logger.info(f"Running iteration {iteration + 1} with seed {seed}")

                # Generate one sample for this iteration
                np.random.seed(seed)
                sample_indices = np.random.randint(0, len(self.data), size=size)
                sample_data = self.data.iloc[sample_indices]

                # Use the same sample for all methods
                for method in self.selection_methods:
                    method_name = method.get_display_name()

                    # Get selected configuration for this method
                    selected_config = method.select_configuration(sample_data)

                    # Calculate performance gap
                    gap = self._calculate_gap(selected_config)
                    method_gaps[method_name].append(gap)

            # After all iterations, calculate and store average gaps and std for each method
            for method_name in method_gaps:
                gaps_array = np.array(method_gaps[method_name])
                self.results[method_name]['gaps'].append(np.mean(gaps_array))
                self.results[method_name]['gap_stds'].append(np.std(gaps_array))
                self.results[method_name]['raw_gaps'].append(gaps_array.tolist())

        # Save results
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        return self.results

    def _calculate_gap(self, config: Dict) -> float:
        """Calculate performance gap between selected and optimal config."""
        # Get score of selected configuration
        mask = pd.Series(True, index=self.rankings.index)
        for col, value in config.items():
            mask &= (self.rankings[col] == value)
        if self.rankings[mask]['mean'].empty:
            return 1.0
        selected_score = self.rankings[mask]['mean'].iloc[0]
        optimal_score = self.rankings['mean'].max()

        return optimal_score - selected_score
