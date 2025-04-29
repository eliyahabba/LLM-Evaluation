"""
Main script for the prompt dimension robustness experiment.
Orchestrates the complete experimental workflow.

Note: This script assumes that data has already been prepared using prepare_data.py.
For analysis only, you can run analysis.py directly.
"""

import argparse
from pathlib import Path

from src.ImproveRobustness.config import ExperimentConfig
from src.ImproveRobustness.prepare_data import setup_directories, load_data
from src.ImproveRobustness.data import prepare_datasets_for_training
from src.ImproveRobustness.training import train_model
from src.ImproveRobustness.evaluation import evaluate_model
from src.ImproveRobustness.analysis import run_analysis


def parse_args():
    """Parse command line arguments with defaults from ExperimentConfig."""
    parser = argparse.ArgumentParser(description="Run prompt dimension robustness experiment")
    
    # Model and data parameters
    parser.add_argument("--model_name", type=str, default=ExperimentConfig.MODEL_NAME, 
                        help="Name of the model to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=ExperimentConfig.BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=ExperimentConfig.LEARNING_RATE, 
                        help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=ExperimentConfig.NUM_TRAIN_EPOCHS, 
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.SEED, 
                        help="Random seed for reproducibility")
    
    # Path parameters
    parser.add_argument("--output_dir", type=str, default=str(ExperimentConfig.OUTPUT_DIR), 
                        help="Directory to save outputs")
    
    # Execution control
    parser.add_argument("--skip_training", action="store_true", 
                        help="Skip the training phase and use existing model")
    parser.add_argument("--skip_evaluation", action="store_true", 
                        help="Skip the evaluation phase")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip the analysis phase")
    parser.add_argument("--analysis_only", action="store_true", 
                        help="Only run the analysis on existing results")
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    args.output_dir = Path(args.output_dir)
    
    # Set random seed
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    return args


def run_experiment():
    """Run the complete experiment pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Define output directories based on args
    output_dir = args.output_dir
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    
    # Handle analysis only mode
    if args.analysis_only:
        print("Running analysis only...")
        run_analysis(output_dir=output_dir, plots_dir=plots_dir)
        return
    
    print("Setting up experiment directories...")
    # Setup directories with output_dir
    data_dir = setup_directories(output_dir)
    
    if not args.skip_training:
        try:
            print("Preparing datasets for training...")
            train_dataset = prepare_datasets_for_training(data_dir=data_dir)
            
            print("Training model...")
            model, tokenizer = train_model(
                train_dataset, 
                model_name=args.model_name,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_train_epochs,
                output_dir=models_dir
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run prepare_data.py first to prepare the data.")
            return
    else:
        print("Loading pre-trained model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(models_dir / "trained_model")
        tokenizer = AutoTokenizer.from_pretrained(models_dir / "trained_model")
    
    if not args.skip_evaluation:
        try:
            print("Loading test data for evaluation...")
            _, test_data = load_data(data_dir=data_dir)
            
            print("Evaluating model...")
            results = evaluate_model(model, tokenizer, test_data, output_dir=output_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run prepare_data.py first to prepare the data.")
            return
    
    print("Experiment completed!")
    
    # Run analysis
    if not args.skip_analysis:
        print("Running analysis...")
        run_analysis(output_dir=output_dir, plots_dir=plots_dir)


if __name__ == "__main__":
    run_experiment() 