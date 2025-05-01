"""
Evaluation script for the dimension robustness experiment.
Evaluates model performance on seen vs. unseen dimension values.
"""

import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_config import ExperimentConfig


class ModelEvaluator:
    def __init__(self, model_path, tokenizer=None, device=None):
        """Initialize the evaluator with a model and tokenizer."""
        self.model_path = Path(model_path)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load tokenizer
        if tokenizer is None:
            print(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.tokenizer = tokenizer

        # Load model
        print(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True
        )

        # Set model to evaluation mode
        self.model.eval()

    def generate_predictions(self, data, batch_size=1):
        """Generate predictions for the given data."""
        # Check if data is DataFrame or path
        if isinstance(data, (str, Path)):
            data = pd.read_parquet(data)

        predictions = []

        # Process in batches
        for i in tqdm(range(0, len(data), batch_size), desc="Generating predictions"):
            batch = data.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                input_text = row["raw_input"]
                ground_truth = row.get("ground_truth", None)
                dimension_value = row.get(ExperimentConfig.TARGET_DIMENSION, None)

                # Tokenize input
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

                # Generate prediction
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=0.1
                    )

                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract the response (remove the input part)
                response = generated_text[len(input_text):].strip()

                # Add prediction to results
                predictions.append({
                    "input": input_text,
                    "prediction": response,
                    "ground_truth": ground_truth,
                    "dimension_value": dimension_value,
                    "is_correct": self._check_correct(response, ground_truth) if ground_truth else None
                })

        # Convert to DataFrame
        return pd.DataFrame(predictions)

    def _check_correct(self, prediction, ground_truth):
        """Check if prediction is correct."""
        # Simple exact match for now - can be made more sophisticated
        return prediction.strip() == ground_truth.strip()

    def evaluate_robustness(self, data_path, output_dir=None, batch_size=1):
        """
        Evaluate model robustness on seen vs. unseen dimension values.

        Args:
            data_path: Path to evaluation data
            output_dir: Directory to save results
            batch_size: Batch size for inference

        Returns:
            Dictionary of evaluation results
        """
        # Set output directory
        if output_dir is None:
            output_dir = ExperimentConfig.RESULTS_DIR / "evaluations"
        else:
            output_dir = Path(output_dir)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"Loading evaluation data from {data_path}")
        data = pd.read_parquet(data_path)

        # Generate predictions
        print("Generating predictions")
        predictions = self.generate_predictions(data, batch_size)

        # Save raw predictions
        predictions_path = output_dir / f"{self.model_path.name}_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

        # Calculate overall accuracy
        accuracy = predictions["is_correct"].mean()

        # Split by dimension values
        seen_values = ExperimentConfig.TRAINING_DIMENSION_VALUES
        unseen_values = ExperimentConfig.EXCLUDED_DIMENSION_VALUES

        seen_preds = predictions[predictions["dimension_value"].isin(seen_values)]
        unseen_preds = predictions[predictions["dimension_value"].isin(unseen_values)]

        # Calculate accuracy for seen and unseen values
        seen_accuracy = seen_preds["is_correct"].mean() if not seen_preds.empty else 0
        unseen_accuracy = unseen_preds["is_correct"].mean() if not unseen_preds.empty else 0
        generalization_gap = seen_accuracy - unseen_accuracy

        # Calculate accuracy for each dimension value
        dimension_accuracies = {}
        for value in set(predictions["dimension_value"].dropna()):
            value_preds = predictions[predictions["dimension_value"] == value]
            dimension_accuracies[value] = value_preds["is_correct"].mean()

        # Organize results
        results = {
            "overall": {
                "accuracy": accuracy,
                "seen_values_accuracy": seen_accuracy,
                "unseen_values_accuracy": unseen_accuracy,
                "generalization_gap": generalization_gap,
                "dimension_variance": np.var(list(dimension_accuracies.values())) if dimension_accuracies else 0
            },
            "by_dimension_value": dimension_accuracies
        }

        # Save results
        self._save_results(results, output_dir)

        return results

    def _save_results(self, results, output_dir):
        """Save evaluation results to file."""
        results_path = output_dir / f"{self.model_path.name}_results.txt"

        with open(results_path, "w") as f:
            f.write("Dimension Robustness Evaluation Results\n")
            f.write("=====================================\n\n")

            f.write("Overall Results:\n")
            for key, value in results["overall"].items():
                f.write(f"  {key}: {value:.4f}\n")

            f.write("\nResults by Dimension Value:\n")
            for value, accuracy in results["by_dimension_value"].items():
                is_seen = value in ExperimentConfig.TRAINING_DIMENSION_VALUES
                status = "SEEN" if is_seen else "UNSEEN"
                f.write(f"  {value} ({status}): {accuracy:.4f}\n")

        print(f"Evaluation results saved to {results_path}")

        # Also save as CSV for easier analysis
        overall_df = pd.DataFrame([results["overall"]])
        overall_path = output_dir / f"{self.model_path.name}_overall_results.csv"
        overall_df.to_csv(overall_path, index=False)

        dimension_df = pd.DataFrame([
            {"dimension_value": value, "accuracy": acc, "is_seen": value in ExperimentConfig.TRAINING_DIMENSION_VALUES}
            for value, acc in results["by_dimension_value"].items()
        ])
        dimension_path = output_dir / f"{self.model_path.name}_dimension_results.csv"
        dimension_df.to_csv(dimension_path, index=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model robustness on dimension values")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model to evaluate")

    # Data arguments
    parser.add_argument("--data_path", type=str, default=str(ExperimentConfig.EVAL_DATA_PATH),
                        help="Path to evaluation data")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default=str(ExperimentConfig.EVAL_DIR),
                        help="Directory to save results")

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for inference (cuda or cpu)")

    args = parser.parse_args()
    return args


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=args.device
    )

    # Evaluate robustness
    evaluator.evaluate_robustness(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()