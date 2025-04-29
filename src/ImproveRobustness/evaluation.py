"""
Evaluation module for the prompt dimension robustness experiment.
Handles model evaluation and metrics calculation.
"""

import torch
import numpy as np
from tqdm import tqdm
from src.ImproveRobustness.config import ExperimentConfig


def evaluate_model(model, tokenizer, test_data, output_dir=None):
    """
    Evaluate model performance on test data, measuring:
    1. Performance on seen vs. unseen dimension values
    2. Variance in performance across dimension values
    """
    if output_dir is None:
        output_dir = ExperimentConfig.OUTPUT_DIR
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group test data by dimension values
    seen_values_data = test_data[test_data[ExperimentConfig.TARGET_DIMENSION].isin(ExperimentConfig.TRAINING_DIMENSION_VALUES)]
    unseen_values_data = test_data[test_data[ExperimentConfig.TARGET_DIMENSION].isin(ExperimentConfig.EXCLUDED_DIMENSION_VALUES)]

    # Group test data by datasets
    dataset_groups = {}
    for dataset in ExperimentConfig.TEST_DATASETS:
        dataset_groups[dataset] = test_data[test_data['dataset'] == dataset]

    results = {
        "overall": {},
        "by_dimension_value": {},
        "by_dataset": {}
    }

    # Evaluate on seen vs. unseen values
    seen_accuracy = evaluate_subset(model, tokenizer, seen_values_data)
    unseen_accuracy = evaluate_subset(model, tokenizer, unseen_values_data)

    results["overall"]["seen_values_accuracy"] = seen_accuracy
    results["overall"]["unseen_values_accuracy"] = unseen_accuracy
    results["overall"]["generalization_gap"] = seen_accuracy - unseen_accuracy

    # Evaluate for each dimension value
    for value in ExperimentConfig.ALL_DIMENSION_VALUES:
        value_data = test_data[test_data[ExperimentConfig.TARGET_DIMENSION] == value]
        if not value_data.empty:
            accuracy = evaluate_subset(model, tokenizer, value_data)
            results["by_dimension_value"][value] = accuracy

    # Calculate variance across dimension values
    values_accuracies = list(results["by_dimension_value"].values())
    results["overall"]["dimension_variance"] = np.var(values_accuracies)

    # Evaluate for each dataset
    for dataset, dataset_data in dataset_groups.items():
        if not dataset_data.empty:
            accuracy = evaluate_subset(model, tokenizer, dataset_data)
            results["by_dataset"][dataset] = accuracy

    # Save results
    with open(output_dir / "evaluation_results.txt", "w") as f:
        f.write("Robustness Experiment Results\n")
        f.write("===========================\n\n")

        f.write("Overall Results:\n")
        for key, value in results["overall"].items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\nResults by Dimension Value:\n")
        for key, value in results["by_dimension_value"].items():
            seen_unseen = "SEEN" if key in ExperimentConfig.TRAINING_DIMENSION_VALUES else "UNSEEN"
            f.write(f"  {key} ({seen_unseen}): {value:.4f}\n")

        f.write("\nResults by Dataset:\n")
        for key, value in results["by_dataset"].items():
            f.write(f"  {key}: {value:.4f}\n")

    return results


def evaluate_subset(model, tokenizer, data_subset):
    """Evaluate model on a subset of data."""
    model.eval()
    correct = 0
    total = 0

    for _, row in tqdm(data_subset.iterrows(), total=len(data_subset), desc="Evaluating"):
        input_text = row["raw_input"]
        ground_truth = row["ground_truth"]

        # Generate prediction
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=False
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = extract_answer(generated_text, input_text)

        # Check if prediction matches ground truth
        if prediction == ground_truth:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def extract_answer(generated_text, input_text):
    """Extract the answer from the generated text."""
    # Remove the input portion
    answer_text = generated_text[len(input_text):].strip()

    # Simple answer extraction (can be improved)
    # This is a placeholder - would need to implement proper extraction logic
    # based on the format of your data
    answer = answer_text.split()[0] if answer_text else ""

    return answer 