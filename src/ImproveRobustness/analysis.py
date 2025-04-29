"""
Analysis module for the prompt dimension robustness experiment.
Handles results analysis and visualization.

This module can be run independently to analyze experiment results.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.ImproveRobustness.config import ExperimentConfig


def load_evaluation_results(results_file=None):
    """Load the evaluation results from the text file."""
    results = {
        "overall": {},
        "by_dimension_value": {},
        "by_dataset": {}
    }

    if results_file is None:
        results_file = ExperimentConfig.OUTPUT_DIR / "evaluation_results.txt"

    if not results_file.exists():
        print(f"Results file not found at {results_file}")
        return None

    current_section = None

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "Overall Results:":
                current_section = "overall"
            elif line == "Results by Dimension Value:":
                current_section = "by_dimension_value"
            elif line == "Results by Dataset:":
                current_section = "by_dataset"
            elif current_section and line.startswith("  "):
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()

                    # Handle the case for dimension values with SEEN/UNSEEN tags
                    if current_section == "by_dimension_value" and "(" in key:
                        key_parts = key.split("(")
                        key = key_parts[0].strip()

                    value = float(parts[1].strip())
                    results[current_section][key] = value

    return results


def plot_dimension_value_performance(results, plots_dir=None):
    """Plot performance across dimension values, highlighting seen vs. unseen."""
    if not results or "by_dimension_value" not in results:
        print("No dimension value results to plot")
        return

    if plots_dir is None:
        plots_dir = ExperimentConfig.PLOTS_DIR
    
    # Create plots directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    dimension_results = results["by_dimension_value"]

    # Prepare data for plotting
    values = []
    accuracies = []
    seen_status = []

    for value, accuracy in dimension_results.items():
        values.append(value)
        accuracies.append(accuracy)
        seen = value in ExperimentConfig.TRAINING_DIMENSION_VALUES
        seen_status.append("Seen during training" if seen else "Unseen during training")

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        "Dimension Value": values,
        "Accuracy": accuracies,
        "Training Status": seen_status
    })

    # Sort by accuracy
    df = df.sort_values("Accuracy", ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Dimension Value", y="Accuracy", hue="Training Status", data=df, palette=["skyblue", "salmon"])
    plt.title(f"Performance Across {ExperimentConfig.TARGET_DIMENSION.split(':')[1].strip()} Values")
    plt.xlabel("Dimension Value")
    plt.ylabel("Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "dimension_value_performance.png")
    plt.close()


def plot_dataset_performance(results, plots_dir=None):
    """Plot performance across different datasets."""
    if not results or "by_dataset" not in results:
        print("No dataset results to plot")
        return

    if plots_dir is None:
        plots_dir = ExperimentConfig.PLOTS_DIR
    
    # Create plots directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_results = results["by_dataset"]

    # Prepare data for plotting
    datasets = list(dataset_results.keys())
    accuracies = list(dataset_results.values())

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    datasets = [datasets[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(datasets, accuracies, color='lightgreen')
    plt.title("Performance Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plots_dir / "dataset_performance.png")
    plt.close()


def plot_generalization_comparison(results, plots_dir=None):
    """Plot the comparison between seen and unseen dimension values."""
    if not results or "overall" not in results:
        print("No overall results to plot")
        return

    if plots_dir is None:
        plots_dir = ExperimentConfig.PLOTS_DIR
    
    # Create plots directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall_results = results["overall"]

    if "seen_values_accuracy" not in overall_results or "unseen_values_accuracy" not in overall_results:
        print("Missing seen/unseen accuracy values")
        return

    # Prepare data for plotting
    categories = ["Seen Values", "Unseen Values"]
    accuracies = [
        overall_results["seen_values_accuracy"],
        overall_results["unseen_values_accuracy"]
    ]

    # Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies, color=['skyblue', 'salmon'])
    plt.title("Generalization to Unseen Dimension Values")
    plt.xlabel("Training Status")
    plt.ylabel("Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add gap annotation
    gap = overall_results.get("generalization_gap", accuracies[0] - accuracies[1])
    plt.annotate(
        f'Gap: {gap:.4f}',
        xy=(0.5, min(accuracies) + (abs(accuracies[0] - accuracies[1]) / 2)),
        xytext=(0.5, min(accuracies) - 0.05),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        ha='center'
    )

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plots_dir / "generalization_comparison.png")
    plt.close()


def generate_summary_table(output_dir=None, data_dir=None):
    """Generate a summary table of the experiment results."""
    if data_dir is None:
        data_dir = ExperimentConfig.DATA_DIR
    
    if output_dir is None:
        output_dir = ExperimentConfig.OUTPUT_DIR
        
    try:
        train_data = pd.read_parquet(data_dir / "train_data.parquet")
        test_data = pd.read_parquet(data_dir / "test_data.parquet")
    except FileNotFoundError:
        print(f"Train or test data not found in {data_dir}")
        return None

    # Count samples by dimension value
    train_counts = train_data[ExperimentConfig.TARGET_DIMENSION].value_counts().to_dict()
    test_counts = test_data[ExperimentConfig.TARGET_DIMENSION].value_counts().to_dict()

    # Count samples by dataset
    train_dataset_counts = train_data['dataset'].value_counts().to_dict()
    test_dataset_counts = test_data['dataset'].value_counts().to_dict()

    # Create summary DataFrame
    summary_data = []

    # Add dimension value counts
    for value in set(list(train_counts.keys()) + list(test_counts.keys())):
        training_status = "Seen" if value in ExperimentConfig.TRAINING_DIMENSION_VALUES else "Unseen"
        summary_data.append({
            "Category": "Dimension Value",
            "Name": value,
            "Training Status": training_status,
            "Training Samples": train_counts.get(value, 0),
            "Testing Samples": test_counts.get(value, 0)
        })

    # Add dataset counts
    for dataset in set(list(train_dataset_counts.keys()) + list(test_dataset_counts.keys())):
        summary_data.append({
            "Category": "Dataset",
            "Name": dataset,
            "Training Status": "-",
            "Training Samples": train_dataset_counts.get(dataset, 0),
            "Testing Samples": test_dataset_counts.get(dataset, 0)
        })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)

    return summary_df


def run_analysis(output_dir=None, plots_dir=None):
    """Run the complete analysis pipeline."""
    # Use ExperimentConfig defaults if not provided
    if output_dir is None:
        output_dir = ExperimentConfig.OUTPUT_DIR
    if plots_dir is None:
        plots_dir = ExperimentConfig.PLOTS_DIR
    
    print("Loading evaluation results...")
    results_file = output_dir / "evaluation_results.txt"
    
    if not results_file.exists():
        print(f"Results file not found at {results_file}")
        return None
    
    results = load_evaluation_results(results_file)

    if results:
        print("Generating visualizations...")
        plot_dimension_value_performance(results, plots_dir)
        plot_dataset_performance(results, plots_dir)
        plot_generalization_comparison(results, plots_dir)

        print("Generating summary table...")
        summary_df = generate_summary_table(output_dir)
        print(summary_df.to_string())

    print("Analysis completed!")
    print(f"Plots saved to {plots_dir}")
    
    return results


def parse_args():
    """Parse command line arguments for standalone analysis."""
    parser = argparse.ArgumentParser(description="Analyze results of prompt dimension robustness experiment")
    
    # Path parameters
    parser.add_argument("--output_dir", type=str, default=str(ExperimentConfig.OUTPUT_DIR), 
                        help="Directory containing evaluation results")
    parser.add_argument("--plots_dir", type=str, default=None,
                        help="Directory to save plots (defaults to OUTPUT_DIR/plots)")
    parser.add_argument("--data_dir", type=str, default=str(ExperimentConfig.DATA_DIR),
                        help="Directory containing processed data files")
    
    # Analysis options
    parser.add_argument("--summary_only", action="store_true",
                        help="Only generate the summary table without plots")
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    args.output_dir = Path(args.output_dir)
    if args.plots_dir:
        args.plots_dir = Path(args.plots_dir)
    else:
        args.plots_dir = args.output_dir / "plots"
    args.data_dir = Path(args.data_dir)
    
    return args


def main():
    """Main function for standalone analysis."""
    args = parse_args()
    
    print(f"Analyzing experiment results from {args.output_dir}")
    print(f"Plots will be saved to {args.plots_dir}")
    
    if args.summary_only:
        print("Generating summary table only...")
        summary_df = generate_summary_table(args.output_dir, args.data_dir)
        if summary_df is not None:
            print(summary_df.to_string())
    else:
        run_analysis(args.output_dir, args.plots_dir)


if __name__ == "__main__":
    main() 