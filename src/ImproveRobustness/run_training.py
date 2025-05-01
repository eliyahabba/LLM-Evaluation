#!/usr/bin/env python
"""
Simple script to run training with default configuration.
"""
from pathlib import Path
from src.ImproveRobustness.config import ExperimentConfig
from src.ImproveRobustness.training import Trainer

def main():
    """Run training with default configuration."""
    print("Starting training with default configuration...")
    
    # Set file paths
    data_dir = ExperimentConfig.DATA_DIR
    train_data_path = data_dir / "train_data.parquet"
    test_data_path = data_dir / "test_data.parquet"
    
    # Check if data files exist
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {train_data_path}")
    if not test_data_path.exists():
        raise FileNotFoundError(f"Testing data file not found: {test_data_path}")
    
    # Print configuration information
    ExperimentConfig.print_config_info()
    
    # Initialize trainer
    trainer = Trainer(
        model_name=ExperimentConfig.MODEL_NAME,
        train_data_path=train_data_path,
        eval_data_path=test_data_path,
        output_dir=ExperimentConfig.RESULTS_DIR,
        access_token=ExperimentConfig.ACCESS_TOKEN,
        quant_config_type=ExperimentConfig.QUANT_CONFIG_TYPE,
        use_lora=ExperimentConfig.USE_LORA,
        target_dimension=ExperimentConfig.TARGET_DIMENSION,
        training_dimension_values=ExperimentConfig.TRAINING_DIMENSION_VALUES,
        excluded_dimension_values=ExperimentConfig.EXCLUDED_DIMENSION_VALUES
    )
    
    # Run the training pipeline
    model_path = trainer.run_pipeline(
        eval_before_finetuning=ExperimentConfig.EVAL_BEFORE_FINETUNING,
        do_finetuning=ExperimentConfig.DO_FINETUNING,
        eval_after_finetuning=ExperimentConfig.EVAL_AFTER_FINETUNING
    )
    
    print(f"Training complete! Model saved to: {model_path}")
    
if __name__ == "__main__":
    # Set random seed for reproducibility
    ExperimentConfig.set_seed()
    main() 