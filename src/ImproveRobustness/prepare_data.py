"""
Data preparation script for the prompt dimension robustness experiment.
This script handles loading raw data and preparing it for training and evaluation.

This script is designed to be run independently before running the main experiment.
"""

import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.ImproveRobustness.config import ExperimentConfig


def setup_directories(output_dir=None):
    """Create necessary directories for the experiment."""
    if output_dir is None:
        # Use default output directory and data directory from config
        ExperimentConfig.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ExperimentConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
        ExperimentConfig.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ExperimentConfig.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        ExperimentConfig.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return ExperimentConfig.DATA_DIR

    # If custom output_dir is provided, create a parallel directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data directory and raw_data subdirectory
    data_dir = output_dir
    raw_data_dir = data_dir / ExperimentConfig.RAW_DATA_DIR.name
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results directory with plots and models subdirectories
    results_dir = output_dir.parent / ExperimentConfig.RESULTS_DIR.name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = results_dir / ExperimentConfig.PLOTS_DIR.name
    models_dir = results_dir / ExperimentConfig.MODELS_DIR.name
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def list_available_files(data_dir):
    """List all available parquet files in the data directory."""
    files = list(data_dir.glob("*.parquet"))
    model_datasets = {}

    for file in files:
        parts = file.stem.split('_', 1)
        if len(parts) == 2:
            model_name, dataset = parts
            if model_name not in model_datasets:
                model_datasets[model_name] = []
            model_datasets[model_name].append(dataset)

    return model_datasets, files


def prepare_data(
    model_name,
    target_dimension,
    training_dimension_values,
    excluded_dimension_values,
    training_datasets,
    test_datasets,
    data_dir=None,
    output_dir=None,
    seed=42,
    force_rebuild=False
):
    """
    Prepare training and testing datasets by:
    1. Including only selected dimension values in training
    2. Creating a balanced dataset across different datasets

    Args:
        model_name (str): Name of the model to use
        target_dimension (str): Target dimension to test robustness for
        training_dimension_values (list): Dimension values to include in training
        excluded_dimension_values (list): Dimension values to exclude from training
        training_datasets (list): Datasets to use for training
        test_datasets (list): Datasets to use for testing
        data_dir (Path, optional): Directory containing raw input data. If None, uses ExperimentConfig.RAW_DATA_DIR
        output_dir (Path): Directory to save processed outputs. If None, uses ExperimentConfig.DATA_DIR
        seed (int): Random seed for reproducibility
        force_rebuild (bool): Force rebuild of datasets even if they already exist

    Returns:
        tuple: Paths to the prepared training and testing data files
    """
    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # Create directories and get the processed data directory
    processed_data_dir = setup_directories(output_dir)

    # Check if processed data already exists
    train_data_path = processed_data_dir / "train_data.parquet"
    test_data_path = processed_data_dir / "test_data.parquet"

    if not force_rebuild and train_data_path.exists() and test_data_path.exists():
        print(f"Using existing processed data from {processed_data_dir}")
        return train_data_path, test_data_path

    # Use RAW_DATA_DIR if data_dir is not provided for raw input data
    if data_dir is None:
        data_dir = ExperimentConfig.RAW_DATA_DIR
    else:
        # Ensure data_dir is a Path object
        data_dir = Path(data_dir)
        # If raw_data is not in the path, look for it in a subdirectory
        if not str(data_dir).endswith("raw_data"):
            raw_data_subdir = data_dir / "raw_data"
            if raw_data_subdir.exists():
                data_dir = raw_data_subdir

    print(f"Using raw data from: {data_dir}")
    print("Processing raw data...")
    model_datasets, files = list_available_files(data_dir)

    # Filter files for our target model
    model_files = [f for f in files if model_name.split('/')[-1] in f.stem]

    def normalize_dataset_name(name):
        return name.replace(".", "_")

    normalized_training_datasets = [normalize_dataset_name(x) for x in training_datasets]
    normalized_test_datasets = [normalize_dataset_name(x) for x in test_datasets]
    all_normalized_datasets = normalized_training_datasets + normalized_test_datasets

    model_files = [
        f for f in model_files
        if "_".join(f.stem.split("_")[1:]) in all_normalized_datasets
    ]

    if not model_files:
        print(f"Warning: No files found for model {model_name}")
        print(f"Available files: {[f.name for f in files]}")
        raise ValueError(f"No data files found for model {model_name} and specified datasets")

    all_dfs = []

    # Load all data files
    for file_path in tqdm(model_files, desc="Loading datasets"):
        df = pd.read_parquet(file_path)
        dataset_name = df['dataset'].iloc[0] if 'dataset' in df.columns else str(file_path.stem).split('_', 1)[1]

        # Add normalized dataset name for easier filtering
        df['normalized_dataset'] = normalize_dataset_name(dataset_name)
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data loaded from the specified files")

    # Combine all data
    all_data = pd.concat(all_dfs)
    print(f"Total data loaded: {len(all_data)} examples")

    # Check if target dimension exists in the data
    if target_dimension not in all_data.columns:
        raise ValueError(f"Target dimension '{target_dimension}' not found in data. Available columns: {all_data.columns.tolist()}")

    # Split into training and test datasets
    train_data = all_data[all_data['normalized_dataset'].isin(normalized_training_datasets)]
    test_data = all_data[all_data['normalized_dataset'].isin(normalized_test_datasets)]

    print(f"Initial training data: {len(train_data)} examples")
    print(f"Initial test data: {len(test_data)} examples")

    # For training data: only include examples with the selected dimension values
    train_data = train_data[train_data[target_dimension].isin(training_dimension_values)]

    # For test data: keep all dimension values to test generalization
    # (both seen and unseen values)

    # Shuffle the data
    train_data = train_data.sample(frac=1, random_state=seed)
    test_data = test_data.sample(frac=1, random_state=seed)

    print(f"Final training data: {len(train_data)} examples")
    print(f"Final test data: {len(test_data)} examples")

    # Check if we have enough data
    if len(train_data) == 0:
        raise ValueError("No training data found with the specified criteria")

    if len(test_data) == 0:
        raise ValueError("No test data found with the specified criteria")

    # Save prepared datasets
    train_data.to_parquet(train_data_path)
    test_data.to_parquet(test_data_path)

    # Save dataset statistics
    save_dataset_statistics(
        train_data,
        test_data,
        target_dimension,
        processed_data_dir
    )

    print(f"Processed data saved to {processed_data_dir}")

    return train_data_path, test_data_path


def save_dataset_statistics(train_data, test_data, target_dimension, processed_data_dir):
    """Save statistics about the datasets for later reference."""
    stats = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_dimension_values": train_data[target_dimension].value_counts().to_dict(),
        "test_dimension_values": test_data[target_dimension].value_counts().to_dict(),
        "train_datasets": train_data['dataset'].value_counts().to_dict() if 'dataset' in train_data.columns else {},
        "test_datasets": test_data['dataset'].value_counts().to_dict() if 'dataset' in test_data.columns else {}
    }

    # Save as CSV for easy viewing
    stats_dir = processed_data_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    # Save dimension value counts
    pd.DataFrame({
        'dimension_value': list(stats['train_dimension_values'].keys()) + list(stats['test_dimension_values'].keys()),
        'split': ['train'] * len(stats['train_dimension_values']) + ['test'] * len(stats['test_dimension_values']),
        'count': list(stats['train_dimension_values'].values()) + list(stats['test_dimension_values'].values())
    }).to_csv(stats_dir / "dimension_value_counts.csv", index=False)

    # Save dataset counts
    pd.DataFrame({
        'dataset': list(stats['train_datasets'].keys()) + list(stats['test_datasets'].keys()),
        'split': ['train'] * len(stats['train_datasets']) + ['test'] * len(stats['test_datasets']),
        'count': list(stats['train_datasets'].values()) + list(stats['test_datasets'].values())
    }).to_csv(stats_dir / "dataset_counts.csv", index=False)

    # Save detailed statistics about dimension values per dataset
    train_stats = train_data.groupby(['dataset', target_dimension]).size().reset_index(name='count')
    test_stats = test_data.groupby(['dataset', target_dimension]).size().reset_index(name='count')

    train_stats['split'] = 'train'
    test_stats['split'] = 'test'

    pd.concat([train_stats, test_stats]).to_csv(
        stats_dir / "dimension_values_by_dataset.csv", index=False
    )

    print(f"Dataset statistics saved to {stats_dir}")


def load_data(data_dir=None):
    """
    Load the prepared training and testing datasets.

    Args:
        data_dir (Path, optional): Directory containing processed data.
            If None, uses ExperimentConfig.DATA_DIR

    Returns:
        tuple: DataFrames containing training and testing data
    """
    if data_dir is None:
        data_dir = ExperimentConfig.DATA_DIR

    train_data_path = data_dir / "train_data.parquet"
    test_data_path = data_dir / "test_data.parquet"

    if not train_data_path.exists() or not test_data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_dir}. "
            f"Please run the prepare_data.py script first to prepare the data."
        )

    train_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)

    return train_data, test_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for prompt dimension robustness experiment")

    # Model and data parameters
    parser.add_argument("--model_name", type=str, default=ExperimentConfig.MODEL_NAME,
                        help="Name of the model to use")
    parser.add_argument("--target_dimension", type=str, default=ExperimentConfig.TARGET_DIMENSION,
                        help="Target dimension to test robustness for")
    parser.add_argument("--training_dimension_values", type=str, nargs="+",
                        default=ExperimentConfig.TRAINING_DIMENSION_VALUES,
                        help="Dimension values to include in training")
    parser.add_argument("--excluded_dimension_values", type=str, nargs="+",
                        default=ExperimentConfig.EXCLUDED_DIMENSION_VALUES,
                        help="Dimension values to exclude from training")
    parser.add_argument("--training_datasets", type=str, nargs="+",
                        default=ExperimentConfig.TRAINING_DATASETS,
                        help="Datasets to use for training")
    parser.add_argument("--test_datasets", type=str, nargs="+",
                        default=ExperimentConfig.TEST_DATASETS,
                        help="Datasets to use for testing")

    # Path parameters
    parser.add_argument("--output_dir", type=str, default=str(ExperimentConfig.DATA_DIR),
                        help="Directory to save processed data (defaults to ExperimentConfig.DATA_DIR)")
    parser.add_argument("--data_dir", type=str, default=str(ExperimentConfig.RAW_DATA_DIR),
                        help="Directory containing raw input data (defaults to ExperimentConfig.RAW_DATA_DIR)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=ExperimentConfig.SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuild of datasets even if they already exist")

    args = parser.parse_args()
    return args


def main():
    """Main function to run data preparation."""
    args = parse_args()

    # Convert string paths to Path objects
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    print("Preparing data for prompt dimension robustness experiment...")
    print(f"Model: {args.model_name}")
    print(f"Target dimension: {args.target_dimension}")
    print(f"Training dimension values: {args.training_dimension_values}")
    print(f"Excluded dimension values: {args.excluded_dimension_values}")
    print(f"Training datasets: {args.training_datasets}")
    print(f"Test datasets: {args.test_datasets}")
    print(f"Raw data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    prepare_data(
        model_name=args.model_name,
        target_dimension=args.target_dimension,
        training_dimension_values=args.training_dimension_values,
        excluded_dimension_values=args.excluded_dimension_values,
        training_datasets=args.training_datasets,
        test_datasets=args.test_datasets,
        data_dir=data_dir,
        output_dir=output_dir,
        seed=args.seed,
        force_rebuild=args.force_rebuild
    )

    print("Data preparation completed!")


if __name__ == "__main__":
    main()