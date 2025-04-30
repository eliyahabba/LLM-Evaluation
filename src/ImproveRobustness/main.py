# """
# Main script for the dimension robustness experiment.
# Orchestrates the complete experimental workflow.
# """
#
# import argparse
# import os
# from pathlib import Path
#
# import torch
# import numpy as np
# import random
#
# from prepare_data import setup_directories, prepare_data
# from evaluate import ModelEvaluator
# from experiment_config import ExperimentConfig
#
# from src.ImproveRobustness.training import Trainer
#
#
# def set_seed(seed):
#     """Set random seeds for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Run the dimension robustness experiment")
#
#     # Input/output
#     parser.add_argument("--input_file", type=str, required=True,
#                         help="Path to input data file")
#     parser.add_argument("--output_dir", type=str, default=str(ExperimentConfig.BASE_DIR),
#                         help="Directory to save results")
#
#     # Model configuration
#     parser.add_argument("--model_name", type=str, default=ExperimentConfig.MODEL_NAME,
#                         help="Name of the model to use")
#     parser.add_argument("--use_lora", action="store_true", default=True,
#                         help="Whether to use LoRA for fine-tuning")
#     parser.add_argument("--quant_config_type", type=str, choices=[None, "4bit", "8bit"], default=None,
#                         help="Quantization configuration type")
#     parser.add_argument("--access_token", type=str, default=None,
#                         help="HuggingFace access token")
#
#     # Dimension parameters
#     parser.add_argument("--target_dimension", type=str, default=ExperimentConfig.TARGET_DIMENSION,
#                         help="Target dimension to test robustness for")
#     parser.add_argument("--training_values", type=str, nargs="+",
#                         default=ExperimentConfig.TRAINING_DIMENSION_VALUES,
#                         help="Dimension values to include in training")
#     parser.add_argument("--excluded_values", type=str, nargs="+",
#                         default=ExperimentConfig.EXCLUDED_DIMENSION_VALUES,
#                         help="Dimension values to exclude from training")
#
#     # Flow control
#     parser.add_argument("--skip_data_prep", action="store_true",
#                         help="Skip data preparation")
#     parser.add_argument("--skip_training", action="store_true",
#                         help="Skip model training")
#     parser.add_argument("--skip_evaluation", action="store_true",
#                         help="Skip model evaluation")
#     parser.add_argument("--eval_only", action="store_true",
#                         help="Only run evaluation on an existing model")
#     parser.add_argument("--existing_model_path", type=str, default=None,
#                         help="Path to existing model for evaluation")
#
#     # Other parameters
#     parser.add_argument("--seed", type=int, default=ExperimentConfig.SEED,
#                         help="Random seed for reproducibility")
#     parser.add_argument("--train_ratio", type=float, default=0.8,
#                         help="Ratio of data to use for training")
#
#     args = parser.parse_args()
#     return args
#
#
# def run_experiment():
#     """Run the complete experiment pipeline."""
#     # Parse arguments
#     args = parse_args()
#
#     # Set random seed
#     set_seed(args.seed)
#
#     # Setup directories
#     dirs = setup_directories(args.output_dir)
#
#     # Set file paths
#     input_file = Path(args.input_file)
#     train_data_path = dirs["data_dir"] / "train_data.parquet"
#     eval_data_path = dirs["data_dir"] / "eval_data.parquet"
#
#     # Prepare data if needed
#     if not args.skip_data_prep and not args.eval_only:
#         print("\n=== Preparing Data ===")
#         prepare_data(
#             input_file=input_file,
#             output_train_file=train_data_path,
#             output_eval_file=eval_data_path,
#             target_dimension=args.target_dimension,
#             training_dimension_values=args.training_values,
#             excluded_dimension_values=args.excluded_values,
#             train_ratio=args.train_ratio,
#             seed=args.seed
#         )
#
#     # Train model if needed
#     model_path = None
#     if not args.skip_training and not args.eval_only:
#         print("\n=== Training Model ===")
#         trainer = Trainer(
#             model_name=args.model_name,
#             train_data_path=train_data_path,
#             eval_data_path=eval_data_path,
#             output_dir=dirs["base_dir"],
#             access_token=args.access_token,
#             quant_config_type=args.quant_config_type,
#             use_lora=args.use_lora,
#             target_dimension=args.target_dimension,
#             training_dimension_values=args.training_values,
#             excluded_dimension_values=args.excluded_values
#         )
#
#         model_path = trainer.run_pipeline(
#             eval_before_finetuning=True,
#             do_finetuning=True,
#             eval_after_finetuning=False  # We'll do our own evaluation
#         )
#
#     # Evaluate model if needed
#     if not args.skip_evaluation:
#         print("\n=== Evaluating Model ===")
#
#         # Use existing model path if specified
#         if args.eval_only and args.existing_model_path:
#             model_path = args.existing_model_path
#         elif args.eval_only and not args.existing_model_path:
#             raise ValueError("Must specify --existing_model_path when using --eval_only")
#
#         evaluator = ModelEvaluator(
#             model_path=model_path
#         )
#
#         evaluator.evaluate_robustness(
#             data_path=eval_data_path,
#             output_dir=dirs["eval_dir"]
#         )
#
#     print("\n=== Experiment Completed! ===")
#     print(f"Results saved to {dirs['base_dir']}")
#
#
# if __name__ == "__main__":
#     run_experiment()