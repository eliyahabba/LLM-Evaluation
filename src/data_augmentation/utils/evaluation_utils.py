import csv
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

from src.data_augmentation.config.constants import (
    SIMPLE_QA_CSV_PATH,
    GSM8K_CSV_PATH,
    RESULTS_ROOT,
    GLOBAL_RANDOM_SEED,
    MAX_ANSWER_TOKENS,
    EVALUATION_ROOT,
    OUTPUT_ROOT
)

# Dataset type constants
DATASET_TYPE_QA = "qa"
DATASET_TYPE_MATH = "math"

# Dataset name to type mapping
DATASET_TYPES = {
    "simple_qa": DATASET_TYPE_QA,
    "gsm8k": DATASET_TYPE_MATH,
    "math": DATASET_TYPE_MATH
}

# Default dataset
DEFAULT_DATASET = "simple_qa"

# Metric name by dataset type
METRIC_BY_DATASET_TYPE = {
    DATASET_TYPE_QA: "avg_f1",
    DATASET_TYPE_MATH: "accuracy"
}

# Default file naming conventions
DEFAULT_METRICS_SUMMARY = "all_models_metrics.json"
DEFAULT_PREDICTIONS_SUFFIX = "_predictions.json"


def get_dataset_type(dataset_name: str) -> str:
    """Get the dataset type from the dataset name."""
    return DATASET_TYPES.get(dataset_name.lower(), DATASET_TYPE_QA)


def get_metric_for_dataset(dataset_name: str) -> str:
    """Get the appropriate metric name for a dataset."""
    dataset_type = get_dataset_type(dataset_name)
    return METRIC_BY_DATASET_TYPE.get(dataset_type, "avg_f1")


def get_evaluation_function(dataset_name: str):
    """Get the appropriate evaluation function for a dataset."""
    dataset_type = get_dataset_type(dataset_name)
    if dataset_type == DATASET_TYPE_MATH:
        return evaluate_gsm8k
    else:
        return evaluate_simple_qa


def extract_answer_from_gsm8k(text: str) -> str:
    """Extracts the numerical answer from GSM8K-formatted text."""
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        number_pattern = r'(-?\d*\.?\d+)'
        number_matches = re.findall(number_pattern, last_line)
        if number_matches:
            return number_matches[-1]
    return ""


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing and removing punctuation."""
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    return " ".join(text.split())


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes F1 score between prediction and ground truth for SimpleQA."""
    prediction_normalized = normalize_text(prediction)
    ground_truth_normalized = normalize_text(ground_truth)

    prediction_tokens = prediction_normalized.split()
    ground_truth_tokens = ground_truth_normalized.split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0

    prediction_counter = Counter(prediction_tokens)
    ground_truth_counter = Counter(ground_truth_tokens)

    common = prediction_counter & ground_truth_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / sum(prediction_counter.values())
    recall = num_common / sum(ground_truth_counter.values())

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_gsm8k(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """Evaluates GSM8K predictions against ground truths."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) doesn't match number of ground truths ({len(ground_truths)})")

    correct = 0
    extracted_pairs = []
    original_responses = []

    for pred, gt in zip(predictions, ground_truths):
        extracted_pred = extract_answer_from_gsm8k(pred)
        extracted_gt = extract_answer_from_gsm8k(gt)

        extracted_pairs.append((extracted_pred, extracted_gt))
        original_responses.append((pred, gt))

        if extracted_pred == extracted_gt and extracted_pred != "":
            correct += 1

    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        "extracted_pairs": extracted_pairs,
        "original_responses": original_responses
    }


def evaluate_simple_qa(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """Evaluates SimpleQA predictions against ground truths using F1 score."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) doesn't match number of ground truths ({len(ground_truths)})")

    f1_scores = []
    original_responses = []

    for pred, gt in zip(predictions, ground_truths):
        f1 = compute_f1_score(pred, gt)
        f1_scores.append(f1)
        original_responses.append((pred, gt))

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "avg_f1": avg_f1,
        "individual_f1_scores": f1_scores,
        "total": len(predictions),
        "original_responses": original_responses
    }


def load_dataset(dataset_path: str, dataset_name: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Load a dataset CSV file into a pandas DataFrame and apply filtering."""
    try:
        df = pd.read_csv(dataset_path)
        original_len = len(df)

        max_tokens = MAX_ANSWER_TOKENS.get(dataset_name)
        if max_tokens is not None:
            if 'answer_tokens' in df.columns:
                mask = df['answer_tokens'].isna() | (df['answer_tokens'] <= max_tokens)
                df = df[mask]
                print(f"Filtered {dataset_name} dataset: {original_len} -> {len(df)} rows (max tokens: {max_tokens})")

        return df
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return pd.DataFrame()


def extract_id_number(id_string: str) -> int:
    """Extract the numeric part from an ID string like "simple_qa_3346"."""
    matches = re.findall(r'\d+', id_string)
    if matches:
        return int(matches[-1])
    return -1


def get_ground_truth_from_dataset(dataset_df: pd.DataFrame, id_num: int, dataset_name: str = DEFAULT_DATASET) -> str:
    """Get the ground truth answer for a given ID from the dataset."""
    try:
        if 0 <= id_num < len(dataset_df):
            dataset_type = get_dataset_type(dataset_name)
            if dataset_type == DATASET_TYPE_MATH:
                return dataset_df.loc[dataset_df["id"] == f"test_{id_num}", "answer"].values[0]
            else:
                return dataset_df.iloc[id_num].get('answer', '')
        else:
            print(f"Warning: ID {id_num} is out of range for the dataset (size: {len(dataset_df)})")
            return ""
    except Exception as e:
        print(f"Error retrieving ground truth for ID {id_num}: {e}")
        return ""


def process_config_file(config_file_path: str, dataset_df: pd.DataFrame, dataset_name: str = DEFAULT_DATASET) -> Dict[
    str, List]:
    """Process a configuration file to extract source IDs for matching with ground truths."""
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        if config_data:
            config_key = list(config_data.keys())[0]
            config_info = config_data[config_key]

            if 'source' in config_info and 'ids' in config_info:
                return {
                    'config_key': config_key,
                    'source': config_info.get('source', []),
                    'ids': config_info.get('ids', [])
                }

        return {}
    except Exception as e:
        print(f"Error processing config file {config_file_path}: {e}")
        return {}


def get_model_name_from_file(filename: str) -> str:
    """Extract the model name from a prediction file name."""
    if filename.endswith(DEFAULT_PREDICTIONS_SUFFIX):
        return filename[:-len(DEFAULT_PREDICTIONS_SUFFIX)]
    return filename.split('.')[0]


def get_all_model_files_in_directory(directory: str) -> List[Tuple[str, str]]:
    """Find all model prediction files in a directory."""
    model_files = []

    try:
        for filename in os.listdir(directory):
            if filename.endswith(DEFAULT_PREDICTIONS_SUFFIX) or filename.endswith('_predictions.json'):
                filepath = os.path.join(directory, filename)
                model_name = get_model_name_from_file(filename)
                model_files.append((model_name, filepath))
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")

    return model_files


def evaluate_model_outputs(
        model_output_path: str = None,
        gt_directory: str = None,
        output_path: str = None,
        simple_qa_path: str = SIMPLE_QA_CSV_PATH,
        gsm8k_path: str = GSM8K_CSV_PATH,
        dataset_name: str = DEFAULT_DATASET,
        model_name: str = None,
        random_seed: int = GLOBAL_RANDOM_SEED,
        export_csv: bool = True,
        export_simple_json: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Evaluate model outputs against ground truth by matching IDs with original datasets."""
    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the datasets
    simple_qa_df = load_dataset(simple_qa_path, "simple_qa")
    gsm8k_df = load_dataset(gsm8k_path, "gsm8k")

    if simple_qa_df.empty and gsm8k_df.empty:
        raise ValueError("Both datasets failed to load. Please check the file paths.")

    # Load the model outputs
    try:
        with open(model_output_path, 'r', encoding='utf-8') as f:
            model_outputs = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading model outputs from {model_output_path}: {e}")

    evaluation_results = {}

    # Process each configuration
    for config_name, predictions in model_outputs.items():
        # Determine dataset type
        config_dataset_type = None
        for dataset_key in DATASET_TYPES.keys():
            if dataset_key in config_name.lower():
                config_dataset_type = get_dataset_type(dataset_key)
                break

        if config_dataset_type is None:
            config_dataset_type = get_dataset_type(dataset_name)

        # Use the appropriate dataset DataFrame
        dataset_df = gsm8k_df if config_dataset_type == DATASET_TYPE_MATH else simple_qa_df

        # Find the corresponding ground truth file
        gt_file_path = None
        for file_name in os.listdir(gt_directory):
            if config_name in file_name and file_name.endswith('.json'):
                gt_file_path = os.path.join(gt_directory, file_name)
                break

        if not gt_file_path:
            print(f"Warning: No ground truth file found for configuration {config_name}")
            continue

        # Process the ground truth file
        gt_info = process_config_file(gt_file_path, dataset_df, dataset_name)

        if not gt_info or 'ids' not in gt_info:
            print(f"Warning: No source IDs found in ground truth file for {config_name}")
            continue

        # Extract ground truths based on IDs
        ground_truths = []
        id_mappings = []

        for id_str in gt_info.get('ids', []):
            id_num = extract_id_number(id_str)
            if id_num >= 0:
                gt = get_ground_truth_from_dataset(dataset_df, id_num, dataset_name)
                ground_truths.append(gt)
                id_mappings.append((id_str, id_num))
            else:
                print(f"Warning: Could not extract ID number from {id_str}")

        # Ensure we have the same number of predictions and ground truths
        if len(predictions) != len(ground_truths):
            print(
                f"Warning: Number of predictions ({len(predictions)}) doesn't match number of ground truths ({len(ground_truths)}) for {config_name}")
            min_length = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_length]
            ground_truths = ground_truths[:min_length]
            id_mappings = id_mappings[:min_length]

        # Evaluate and store results
        evaluation_func = get_evaluation_function(dataset_name)
        results = evaluation_func(predictions, ground_truths)
        results['id_mappings'] = id_mappings
        evaluation_results[config_name] = results

    # Save the evaluation results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2)

        if export_csv:
            export_results_to_csv(evaluation_results, output_path, dataset_name)

        if export_simple_json:
            export_results_to_simple_json(evaluation_results, output_path, model_name, dataset_name)

    return evaluation_results


def export_results_to_csv(evaluation_results: Dict[str, Dict[str, Any]], output_path: str,
                          dataset_name: str = DEFAULT_DATASET) -> None:
    """Export evaluation results to a CSV file with a simplified format."""
    if output_path.endswith('.json'):
        csv_path = output_path.replace('.json', '.csv')
    else:
        csv_path = f"{output_path}.csv"

    csv_data = []
    for config_name, metrics in evaluation_results.items():
        extracted_pairs = metrics.get("extracted_pairs", [])
        original_responses = metrics.get("original_responses", [])

        total_items = max(len(extracted_pairs), len(original_responses))

        if total_items > 0:
            for i in range(total_items):
                item_data = {
                    "configuration": config_name,
                    "item_index": i
                }

                if extracted_pairs and i < len(extracted_pairs):
                    pred, gt = extracted_pairs[i]
                    item_data["gt"] = gt
                    item_data["model response after post process"] = pred
                elif original_responses and i < len(original_responses):
                    pred, gt = original_responses[i]
                    item_data["gt"] = gt
                    item_data["model response after post process"] = pred

                csv_data.append(item_data)
        else:
            csv_data.append({
                "configuration": config_name,
                "item_index": -1,
                "gt": "",
                "model response after post process": ""
            })

    csv_data.sort(key=lambda x: (x["configuration"], x["item_index"]))

    fieldnames = ["configuration", "item_index", "gt", "model response after post process"]

    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"CSV results saved to: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV results: {e}")


def export_results_to_simple_json(evaluation_results: Dict[str, Dict[str, Any]], output_path: str, model_name: str,
                                  dataset_name: str = DEFAULT_DATASET) -> None:
    """Export evaluation results to a simplified JSON format."""
    metric_name = get_metric_for_dataset(dataset_name)

    if output_path.endswith('.json'):
        simple_json_path = output_path.replace('.json', '.simple.json')
    else:
        simple_json_path = f"{output_path}.simple.json"

    simple_data = {model_name: {}}

    for config_name, metrics in evaluation_results.items():
        metric_value = metrics.get(metric_name, 0.0)
        simple_data[model_name][config_name] = metric_value

    try:
        os.makedirs(os.path.dirname(simple_json_path), exist_ok=True)
        with open(simple_json_path, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, indent=2)
        print(f"Simplified JSON results saved to: {simple_json_path}")
    except Exception as e:
        print(f"Error saving simplified JSON results: {e}")


def main():
    """Main function to run the evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth")
    parser.add_argument("--model-outputs", default=RESULTS_ROOT,
                        help="Path to the model outputs file or directory.")
    parser.add_argument("--gt-directory", default=OUTPUT_ROOT,
                        help="Directory containing ground truth configuration files.")
    parser.add_argument("--output", default=EVALUATION_ROOT,
                        help="Path to save evaluation results.")
    parser.add_argument("--simple-qa", default=SIMPLE_QA_CSV_PATH, help="Path to the SimpleQA dataset.")
    parser.add_argument("--gsm8k", default=GSM8K_CSV_PATH, help="Path to the GSM8K dataset.")
    parser.add_argument("--dataset", default="simple_qa", choices=list(DATASET_TYPES.keys()), help="Dataset name.")
    parser.add_argument("--model", default="Llama-3.3-70B", help="Model name.")
    parser.add_argument("--seed", type=int, default=GLOBAL_RANDOM_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV export of results.")
    parser.add_argument("--no-simple-json", action="store_true", help="Disable simplified JSON export of results.")

    args = parser.parse_args()

    # Construct paths based on dataset
    model_outputs_path = args.model_outputs / args.dataset if isinstance(args.model_outputs, Path) else Path(
        args.model_outputs) / args.dataset
    gt_directory_path = args.gt_directory / args.dataset / "data" if isinstance(args.gt_directory, Path) else Path(
        args.gt_directory) / args.dataset / "data"
    output_path = args.output / args.dataset if isinstance(args.output, Path) else Path(args.output) / args.dataset

    # Single model or directory of models
    if os.path.isdir(model_outputs_path):
        model_files = get_all_model_files_in_directory(model_outputs_path)

        if not model_files:
            print(f"No model prediction files found in {model_outputs_path}")
            return

        print(f"Found {len(model_files)} model prediction files to evaluate")
        os.makedirs(output_path, exist_ok=True)

        summary = {}

        for model_name, filepath in model_files:
            try:
                print(f"Evaluating {model_name} on {args.dataset}...")
                eval_output_path = os.path.join(output_path, f"{model_name}_evaluation.json")

                model_results = evaluate_model_outputs(
                    model_output_path=filepath,
                    gt_directory=gt_directory_path,
                    output_path=eval_output_path,
                    simple_qa_path=args.simple_qa,
                    gsm8k_path=args.gsm8k,
                    dataset_name=args.dataset,
                    model_name=model_name,
                    random_seed=args.seed,
                    export_csv=not args.no_csv,
                    export_simple_json=not args.no_simple_json
                )

                # Create summary metrics
                summary[model_name] = {}
                metric_name = get_metric_for_dataset(args.dataset)

                for config_name, metrics in model_results.items():
                    if metric_name in metrics:
                        if metric_name not in summary[model_name]:
                            summary[model_name][metric_name] = {}
                        summary[model_name][metric_name][config_name] = metrics[metric_name]

                # Print summary
                print(f"\nResults for {model_name}:")
                for config_name, metrics in model_results.items():
                    if "accuracy" in metrics:
                        print(
                            f"  {config_name} - Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
                    elif "avg_f1" in metrics:
                        print(f"  {config_name} - Average F1: {metrics['avg_f1']:.4f} (Total: {metrics['total']})")

            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")

        # Save summary
        summary_path = os.path.join(output_path, DEFAULT_METRICS_SUMMARY)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary metrics saved to: {summary_path}")

    else:
        # Process a single model file
        evaluation_results = evaluate_model_outputs(
            model_outputs_path,
            gt_directory_path,
            output_path,
            args.simple_qa,
            args.gsm8k,
            args.dataset,
            args.model,
            args.seed,
            export_csv=not args.no_csv,
            export_simple_json=not args.no_simple_json
        )

        # Print summary
        for config_name, metrics in evaluation_results.items():
            print(f"\nResults for {config_name}:")
            if "accuracy" in metrics:
                print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            elif "avg_f1" in metrics:
                print(f"  Average F1: {metrics['avg_f1']:.4f} (Total: {metrics['total']})")


if __name__ == "__main__":
    main()
