# experiment_runner.py
import pandas as pd
import numpy as np
import json
import logging
import os
from typing import List, Dict
from selection_methods import SelectionMethod
from constants import EXPERIMENT_CONFIG


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
            sample_sizes: List[int] = EXPERIMENT_CONFIG['SAMPLE_SIZES'],
            n_iterations: int = EXPERIMENT_CONFIG['N_ITERATIONS'],
            random_seed: int = EXPERIMENT_CONFIG['RANDOM_SEED']
    ) -> Dict:
        """Run experiments for different sample sizes and methods."""
        np.random.seed(random_seed)

        for method in self.selection_methods:
            method_name = method.__class__.__name__
            self.results[method_name] = {
                'sample_sizes': sample_sizes,
                'gaps': []
            }

            logger = logging.getLogger(__name__)
            for size in sample_sizes:
                logger.info(f"Processing sample size: {size}")
                gaps = []
                for iteration in range(n_iterations):
                    # Select random sample
                    sample_indices = np.random.choice(
                        len(self.data),
                        size=size,
                        replace=True
                    )
                    sample_data = self.data.iloc[sample_indices]

                    # Get selected configuration
                    selected_config = method.select_configuration(sample_data)

                    # Calculate performance gap
                    gap = self._calculate_gap(selected_config)
                    gaps.append(gap)

                # Store average gap for this sample size
                self.results[method_name]['gaps'].append(np.mean(gaps))

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
