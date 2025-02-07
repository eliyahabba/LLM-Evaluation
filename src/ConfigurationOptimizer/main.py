# main.py
import logging
import os
from typing import List, Optional

import numpy as np

from data_processor import DataProcessor
from configuration_evaluator import ConfigurationEvaluator
from selection_methods import MajoritySelection, AxiswiseSelection, RegressionBasedSelection, RandomSelection
from experiment_runner import ExperimentRunner
from visualization import Visualizer
from utils import setup_logging
from constants import get_file_paths, ExperimentConfig, DEFAULT_CONFIG, ZERO_SHOT_CONFIG, MODELS
from constants import EXPERIMENT_ARGS_CONFIG


def process_model(
        model_name: str,
        experiment_config: ExperimentConfig,
        force_recompute: bool = True,
        logger: Optional[logging.Logger] = None
) -> None:
    """Process a single model's data."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Starting processing for model: {model_name}")
    logger.info(f"{'=' * 50}")

    # Get file paths for this model and experiment configuration
    paths = get_file_paths(model_name, experiment_config)

    # Initialize components
    data_processor = DataProcessor(paths['input'], experiment_config)
    data = data_processor.load_and_clean_data()

    # Apply experiment-specific filters
    filtered_data = experiment_config.apply_filters(data)
    logger.info(
        f"Applied filters: {experiment_config.filters}. "
        f"Remaining samples: {len(filtered_data)}"
    )
    logger.info(f"Model {model_name}: Loaded data with shape: {filtered_data.shape}")

    evaluator = ConfigurationEvaluator(filtered_data, experiment_config, paths['rankings'])
    rankings = evaluator.compute_rankings(force_recompute)
    if experiment_config.filters is not None and 0 in experiment_config.filters.values():
        rankings = rankings[rankings['count'] > 400]
    else:
        rankings = rankings[rankings['count'] > 1000]

    # Convert to numpy arrays for faster operations
    valid_arr = rankings[['shots', 'template', 'separator', 'enumerator', 'choices_order']].values
    filter_arr = filtered_data[['shots', 'template', 'separator', 'enumerator', 'choices_order']].values

    # Create a fast lookup using structured arrays
    valid_combinations = set(map(tuple, valid_arr))

    # Use boolean indexing with numpy
    mask = np.array([tuple(row) in valid_combinations for row in filter_arr])
    filtered_data = filtered_data[mask]

    logger.info(f"Model {model_name}: Computed rankings for {len(rankings)} configurations")

    # Setup selection methods
    selection_methods = [
        MajoritySelection(experiment_config,rankings),
        AxiswiseSelection(experiment_config,rankings),
        RandomSelection(experiment_config,rankings),
        RegressionBasedSelection(experiment_config,rankings)
    ]

    # Run experiments
    runner = ExperimentRunner(filtered_data, rankings, selection_methods, paths['results'])
    # Experiment configuration
    experiment_args_config =  EXPERIMENT_ARGS_CONFIG
    max_samples = len(filtered_data)
    adjusted_sizes = [size for size in experiment_args_config['SAMPLE_SIZES'] if size <= max_samples]
    # add the max_samples to the list of adjusted sizes if it is not already there
    if max_samples not in adjusted_sizes:
        adjusted_sizes.append(max_samples)
    if len(adjusted_sizes) < len(experiment_args_config['SAMPLE_SIZES']):
        logger.warning(
            f"Adjusted sample sizes due to data size ({max_samples} samples). "
            f"Using sizes: {adjusted_sizes}"
        )
    # update the experiment_args_config with the adjusted sizes
    experiment_args_config['SAMPLE_SIZES'] = adjusted_sizes

    results = runner.run_experiments(experiment_args_config['SAMPLE_SIZES'], experiment_args_config['N_ITERATIONS'], experiment_args_config['RANDOM_SEED'])
    logger.info(f"Model {model_name}: Completed experiments")

    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_performance_gaps(
        results,
        paths['plot'].replace('.png', f'{experiment_config.experiment_suffix}.png'),
        model_name
    )
    logger.info(f"Model {model_name}: Generated visualization")


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Define experiment configurations
    experiments = [
        # DEFAULT_CONFIG,
        ZERO_SHOT_CONFIG
    ]
    # Model configurations
    MODELS = [
        'meta-llama/Llama-3.2-1B-Instruct',
    ]
    # Process each model with each experiment configuration
    for model_name in MODELS:
        for experiment_config in experiments:
            try:
                logger.info(
                    f"Starting experiment with config: "
                    f"{experiment_config.experiment_suffix or 'default'}"
                )
                process_model(
                    model_name=model_name,
                    experiment_config=experiment_config,
                    logger=logger
                )
                logger.info(
                    f"Successfully processed model {model_name} "
                    f"with config {experiment_config.experiment_suffix or 'default'}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing model {model_name} "
                    f"with config {experiment_config.experiment_suffix or 'default'}: "
                    f"{str(e)}"
                )
                continue


if __name__ == "__main__":
    main()