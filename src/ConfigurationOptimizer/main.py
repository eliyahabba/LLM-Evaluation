# main.py
import logging
from typing import Optional

from configuration_evaluator import ConfigurationEvaluator
from constants import MODELS, get_file_paths
from data_processor import DataProcessor
from experiment_runner import ExperimentRunner
from selection_methods import MajoritySelection, AxiswiseSelection
from utils import setup_logging
from visualization import Visualizer


def process_model(
        model_name: str,
        force_recompute: bool = False,
        logger: Optional[logging.Logger] = None
) -> None:
    """Process a single model's data."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Starting processing for model: {model_name}")
    logger.info(f"{'=' * 50}")

    # Get file paths for this model
    paths = get_file_paths(model_name)

    # Initialize components
    data_processor = DataProcessor(paths['input'])
    data = data_processor.load_and_clean_data()
    logger.info(f"Model {model_name}: Loaded data with shape: {data.shape}")

    evaluator = ConfigurationEvaluator(data, paths['rankings'])
    rankings = evaluator.compute_rankings(force_recompute)
    logger.info(f"Model {model_name}: Computed rankings for {len(rankings)} configurations")

    # Setup selection methods
    selection_methods = [
        MajoritySelection(),
        AxiswiseSelection()
    ]

    # Run experiments
    runner = ExperimentRunner(data, rankings, selection_methods, paths['results'])
    results = runner.run_experiments()
    logger.info(f"Model {model_name}: Completed experiments")

    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_performance_gaps(
        results,
        paths['plot'],
        model_name
    )
    logger.info(f"Model {model_name}: Generated visualization")


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Process each model
    for model_name in MODELS:
        try:
            process_model(model_name, logger=logger)
            logger.info(f"Successfully processed model: {model_name}")
        except Exception as e:
            logger.error(f"Error processing model {model_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
