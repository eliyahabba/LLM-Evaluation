"""
Main script for the data augmentation system.
"""

import os
import argparse
import logging
import random
import numpy as np
from pathlib import Path
from src.data_augmentation.utils.data_downloader import DataDownloader
from src.data_augmentation.utils.config_generator import ConfigGenerator
from src.data_augmentation.utils.data_processor import DataProcessor
from src.data_augmentation.config.constants import (
    DATASETS,
    OUTPUT_ROOT,
    MAX_CONFIGS_PER_DATASET,
    EXAMPLES_PER_CONFIG,
    GLOBAL_RANDOM_SEED
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_augmentation.log')
        ]
    )
    return logging.getLogger(__name__)


def initialize_random_seeds(seed=GLOBAL_RANDOM_SEED):
    """Initialize random seeds for all relevant modules for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    # Add any other random modules that might need seeding
    return seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data Augmentation System for LM Testing')
    
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        choices=DATASETS.keys(),
        default=list(DATASETS.keys()),
        help='Datasets to process'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=str(OUTPUT_ROOT),
        help='Root directory for output files'
    )
    
    parser.add_argument(
        '--cache_dir', 
        type=str, 
        default=None,
        help='Directory to cache downloaded datasets'
    )
    
    parser.add_argument(
        '--limit_configs', 
        type=int, 
        default=None,
        help='Limit the number of configurations to generate per dataset'
    )
    
    parser.add_argument(
        '--examples_per_config',
        type=int,
        default=EXAMPLES_PER_CONFIG,
        help='Number of examples to process per configuration (overrides constant)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=GLOBAL_RANDOM_SEED,
        help='Random seed for deterministic behavior'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the data augmentation pipeline."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting data augmentation process")
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize random seeds
    used_seed = initialize_random_seeds(args.random_seed)
    logger.info(f"Initialized random seeds with value: {used_seed}")
    
    logger.info(f"Processing datasets: {args.datasets}")
    logger.info(f"Examples per configuration: {args.examples_per_config}")
    logger.info(f"Max configurations per dataset: {MAX_CONFIGS_PER_DATASET}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Override OUTPUT_ROOT if custom output directory is specified
    if args.output_dir != str(OUTPUT_ROOT):
        # Import and modify the constants module directly
        import src.data_augmentation.config.constants as constants
        
        # Update OUTPUT_ROOT to use the custom directory
        constants.OUTPUT_ROOT = Path(args.output_dir)
        logger.info(f"Updated output root to: {constants.OUTPUT_ROOT}")
        
        # Also update dataset output paths
        for dataset in args.datasets:
            constants.DATASET_OUTPUT[dataset] = constants.OUTPUT_ROOT / dataset
            
        # Log the new paths
        for dataset in args.datasets:
            logger.info(f"  - Output dir for {dataset}: {constants.DATASET_OUTPUT[dataset]}")
            logger.info(f"  - Config dir: {constants.get_config_dir(dataset)}")
            logger.info(f"  - Data dir: {constants.get_data_dir(dataset)}")
    
    # Update examples per config if specified in arguments
    if args.examples_per_config != EXAMPLES_PER_CONFIG:
        import src.data_augmentation.config.constants as constants
        constants.EXAMPLES_PER_CONFIG = args.examples_per_config
        logger.info(f"Updated examples per config to: {constants.EXAMPLES_PER_CONFIG}")
    
    # Initialize components
    downloader = DataDownloader(cache_dir=args.cache_dir)
    config_generator = ConfigGenerator()
    data_processor = DataProcessor()
    
    # Download and prepare datasets
    all_examples = {}
    for dataset_name in args.datasets:
        logger.info(f"Downloading and preparing dataset: {dataset_name}")
        examples = downloader.download_and_prepare(dataset_name)
        all_examples[dataset_name] = examples
        logger.info(f"Downloaded {len(examples)} examples for {dataset_name}")
    
    # Generate configurations for each dataset
    logger.info("Generating configurations")
    all_configs = config_generator.generate_all_configs()
    
    # Apply limit to configurations if specified
    if args.limit_configs is not None:
        for dataset in all_configs:
            all_configs[dataset] = all_configs[dataset][:args.limit_configs]
        logger.info(f"Limited to {args.limit_configs} configurations per dataset")
    
    # Save configuration files
    config_files = config_generator.save_configs(all_configs)
    
    # Count total configs
    total_configs = sum(len(configs) for configs in all_configs.values())
    logger.info(f"Generated and saved {total_configs} configurations in total")
    
    # Process datasets according to configurations
    logger.info("Processing data according to configurations")
    processed_files = data_processor.process_configs(all_examples, all_configs)
    
    # Count total processed files
    total_files = sum(len(files) for files in processed_files.values())
    logger.info(f"Processed and saved {total_files} data files in total")
    
    # Print the locations of the files
    for dataset, files in processed_files.items():
        if files:
            logger.info(f"Files generated for dataset {dataset}:")
            for file in files:
                logger.info(f"  - {file}")
    
    logger.info("Data augmentation process completed")


if __name__ == "__main__":
    main() 