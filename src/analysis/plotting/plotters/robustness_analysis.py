#!/usr/bin/env python3
"""
Improved Per-Question Robustness Analysis

Creates per-question robustness analysis showing how consistently models answer
individual questions correctly across different prompt configurations.
This analysis helps identify which questions are robust to prompt variations
and which are sensitive to prompt formatting.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend - no display windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.analysis.plotting.utils.config import (
    DEFAULT_MODELS, DEFAULT_DATASETS, DEFAULT_SHOTS, PLOT_STYLE,
    get_model_display_name
)
from src.analysis.plotting.utils.data_manager import DataManager
from src.analysis.plotting.utils.auth import ensure_hf_authentication
from src.analysis.plotting.utils.config import get_cache_directory


class RobustnessAnalyzer:
    """
    Creates per-question robustness analysis plots showing success rate distributions.
    
    This class generates histograms showing how many questions have different levels
    of robustness (consistency) across various prompt configurations.
    """

    def __init__(self):
        """Initialize the analyzer with consistent plot styling from config."""
        plt.rcParams['font.family'] = PLOT_STYLE['font_family']
        plt.rcParams['font.serif'] = PLOT_STYLE['font_serif']
        plt.rcParams['figure.dpi'] = PLOT_STYLE['figure_dpi']

    def _aggregate_samples_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-question statistics: how many prompt configurations answered correctly,
        total attempts, and success percentage.
        
        Args:
            df: DataFrame with raw evaluation data
            
        Returns:
            DataFrame with columns:
            - dataset
            - sample_index  
            - sum_correct
            - total_configurations
            - fraction_correct
        """
        print("  Computing per-question accuracy statistics...")
        start_time = time.time()

        # Group by dataset and question index - working with 'score' instead of 'accuracy'
        grouped = df.groupby(["dataset", "sample_index"])["score"]

        # Count correct answers and total attempts
        sum_correct = grouped.sum()
        total_configurations = grouped.count()

        # Create DataFrame with statistics
        question_stats = pd.DataFrame({
            "sum_correct": sum_correct,
            "total_configurations": total_configurations
        }).reset_index()

        # Calculate success percentage
        question_stats["fraction_correct"] = (
                question_stats["sum_correct"] / question_stats["total_configurations"]
        ).round(3)

        print(f"  ✓ Aggregation completed in {time.time() - start_time:.2f} seconds")
        return question_stats

    def _create_robustness_histogram(
            self,
            question_stats: pd.DataFrame,
            model_name: str,
            output_dir: str = "plots",
            force_overwrite: bool = False
    ):
        """
        Create robustness histogram showing success rate distribution across all questions.
        
        Args:
            question_stats: DataFrame with per-question statistics
            model_name: Name of the model being analyzed
            output_dir: Directory to save plots
            force_overwrite: Whether to overwrite existing files
        """
        if question_stats.empty:
            print(f"⚠️  No question statistics for {model_name} - skipping plot")
            return

        print(f"📊 Creating per-question robustness histogram for {model_name}")
        print(f"  Total questions: {len(question_stats)}")

        # File settings
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        filename = f"robustness_histogram"
        model_output_dir = f'{output_dir}/{safe_model_name}'
        output_png = f'{model_output_dir}/{filename}.png'

        # Create the plot - exactly like the original file
        fig = plt.figure(figsize=(18, 12.5))
        ax = plt.gca()

        # White background (not transparent) and remove frames
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Convert to percentages
        percentage_correct = question_stats['fraction_correct'] * 100
        bins = np.arange(0, 105, 5)  # Up to 105 to include 100 in the last bin

        # Create histogram
        n, bins, patches = plt.hist(
            percentage_correct,
            bins=bins,
            edgecolor='black',
            color='skyblue'
        )

        # Color the bars by values - like in the original file
        for i in range(len(patches)):
            bin_center = (bins[i] + bins[i + 1]) / 2
            if bin_center >= 90:
                patches[i].set_facecolor('green')
            elif bin_center <= 10:
                patches[i].set_facecolor('red')
            else:
                patches[i].set_facecolor('skyblue')

        # Add number labels above each bar
        for i in range(len(patches)):
            plt.text(
                patches[i].get_x() + patches[i].get_width() / 2,
                patches[i].get_height(),
                f'{int(n[i])}',
                ha='center',
                va='bottom',
                color='darkblue',
                fontsize=18,
                fontweight='bold'
            )

        # Axis labels
        plt.xlabel('Success Rate Over Different Prompts (%)', fontsize=32)
        plt.ylabel('Number of Questions in Dataset', fontsize=32)

        # Set X-axis range
        plt.xlim(0, 100)

        # Increase font size for labels
        plt.xticks(fontsize=24)
        plt.yticks([])  # Remove Y-axis labels

        # Calculate total number of questions (sum of all bars)
        total_questions = len(question_stats)

        # Title with model name and total question count
        model_display = get_model_display_name(model_name).replace('\n', '')
        plt.title(f'{model_display}\nPer-Question Robustness Analysis\n(Total Questions: {total_questions})',
                  fontsize=28, fontfamily='DejaVu Serif', pad=20)

        plt.tight_layout()

        # Create directory for model
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)

        # Save
        plt.savefig(
            output_png,
            dpi=PLOT_STYLE['figure_dpi'],
            bbox_inches=PLOT_STYLE['bbox_inches'],
            transparent=PLOT_STYLE['transparent'],
            facecolor=PLOT_STYLE['facecolor']
        )
        plt.savefig(
            f'{model_output_dir}/{filename}.svg',
            bbox_inches=PLOT_STYLE['bbox_inches'],
            transparent=PLOT_STYLE['transparent'],
            facecolor=PLOT_STYLE['facecolor']
        )

        plt.close()

        # Statistics
        total_questions = len(question_stats)
        high_robustness = len(question_stats[question_stats['fraction_correct'] >= 0.9])
        low_robustness = len(question_stats[question_stats['fraction_correct'] <= 0.1])

        print(f"✅ Created robustness histogram for {model_name}")
        print(f"  - Total questions: {total_questions}")
        print(f"  - High robustness (≥90%): {high_robustness} ({high_robustness / total_questions * 100:.1f}%)")
        print(f"  - Low robustness (≤10%): {low_robustness} ({low_robustness / total_questions * 100:.1f}%)")

    def create_robustness_analysis(
            self,
            model_name: str,
            datasets: List[str],
            shots_list: List[int] = [0, 5],
            data_manager: DataManager = None,
            output_dir: str = "plots",
            force_overwrite: bool = False
    ):
        """
        Perform robustness analysis for each question for one model across all datasets.
        
        Args:
            model_name: Name of the model to analyze
            datasets: List of datasets to include in analysis
            shots_list: List of shot counts to analyze
            data_manager: DataManager instance for loading data
            output_dir: Directory to save plots
            force_overwrite: Whether to overwrite existing files
        """
        print(f"🔍 Starting per-question robustness analysis for {model_name}")
        print(f"  Datasets: {len(datasets)}")
        print(f"  Shots: {shots_list}")

        all_question_stats = []

        # Process each dataset separately (for memory efficiency)
        for dataset in tqdm(datasets, desc=f"Processing datasets for {model_name}"):
            try:
                # Load raw data for this dataset only - not aggregated!
                dataset_data = data_manager.load_multiple_models(
                    model_names=[model_name],
                    datasets=[dataset],
                    shots_list=shots_list,
                    aggregate=False,  # Important change - raw data!
                    num_processes=1  # No need for parallelism here
                )

                if dataset_data.empty:
                    print(f"⚠️  No data for {model_name} on {dataset} - skipping")
                    continue

                # Filter data like in the original file
                # Only 0-shot or without correct_first/correct_last in 5-shot
                filtered_data = dataset_data[
                    (dataset_data['dimensions_5: shots'] == 0) |
                    (~dataset_data['dimensions_3: choices_order'].isin(["correct_first", "correct_last"]))
                    ]

                if filtered_data.empty:
                    print(f"⚠️  No valid data after filtering for {model_name} on {dataset}")
                    continue

                # Add sample_index if not present
                if 'sample_index' not in filtered_data.columns:
                    filtered_data = filtered_data.reset_index()
                    if 'sample_index' not in filtered_data.columns:
                        filtered_data['sample_index'] = range(len(filtered_data))

                # Compute per-question statistics
                dataset_question_stats = self._aggregate_samples_accuracy(filtered_data)

                if not dataset_question_stats.empty:
                    all_question_stats.append(dataset_question_stats)
                    print(f"  ✓ {dataset}: {len(dataset_question_stats)} questions processed")

            except Exception as e:
                print(f"⚠️  Error processing {dataset} for {model_name}: {e}")
                continue

        if not all_question_stats:
            print(f"❌ No valid question statistics found for {model_name}")
            return

        # Combine all statistics from all datasets
        combined_question_stats = pd.concat(all_question_stats, ignore_index=True)

        # Filter questions with minimum number of configurations (like in original file)
        min_configurations = 99  # Can be adjusted
        valid_questions = combined_question_stats[
            combined_question_stats['total_configurations'] >= min_configurations
            ]

        print(f"  📊 Final statistics:")
        print(f"    - Total questions before filtering: {len(combined_question_stats)}")
        print(f"    - Valid questions (≥{min_configurations} configs): {len(valid_questions)}")

        if valid_questions.empty:
            print(f"❌ No questions with sufficient configurations for {model_name}")
            return

        # Create the plot
        self._create_robustness_histogram(
            question_stats=valid_questions,
            model_name=model_name,
            output_dir=output_dir,
            force_overwrite=force_overwrite
        )


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description='Create per-question robustness analysis plots for LLM evaluation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_robustness_analysis.py
  python run_robustness_analysis.py --models meta-llama/Llama-3.2-1B-Instruct
  python run_robustness_analysis.py --datasets ai2_arc.arc_challenge hellaswag
  python run_robustness_analysis.py --num-processes 1 --no-cache
        """
    )

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'List of models to analyze (default: {len(DEFAULT_MODELS)} models)')

    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help=f'List of datasets to analyze (default: {len(DEFAULT_DATASETS)} datasets)')

    parser.add_argument('--shots', nargs='+', type=int, default=DEFAULT_SHOTS,
                        help=f'List of shot counts to analyze (default: {DEFAULT_SHOTS})')

    parser.add_argument('--num-processes', type=int, default=1,
                        help='Number of parallel processes (default: 1 - sequential processing)')

    parser.add_argument('--output-dir', default="plots/success_rate_distribution",
                        help='Output directory for plots (default: plots/success_rate_distribution)')

    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (default: cache enabled)')

    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing plot files (default: skip existing)')

    parser.add_argument('--list-models', action='store_true',
                        help='List available default models and exit')

    parser.add_argument('--list-datasets', action='store_true',
                        help='List available default datasets and exit')

    return parser.parse_args()


def main():
    """
    Main function - runs the analysis for all models.
    
    Creates per-question robustness histograms showing how consistently each model
    answers individual questions correctly across different prompt configurations.
    """
    args = parse_arguments()

    # Handle list options
    if args.list_models:
        print("Default models:")
        for i, model in enumerate(DEFAULT_MODELS, 1):
            print(f"  {i}. {model}")
        return

    if args.list_datasets:
        print("Default datasets:")
        for i, dataset in enumerate(DEFAULT_DATASETS, 1):
            print(f"  {i}. {dataset}")
        return

    # Perform HuggingFace authentication once at the start
    print("🔐 Authenticating with HuggingFace...")
    ensure_hf_authentication()

    models_to_evaluate = args.models
    selected_datasets = args.datasets
    shots_to_evaluate = args.shots
    num_processes = args.num_processes
    use_cache = not args.no_cache
    output_dir = args.output_dir
    force_overwrite = args.force

    print("Starting Per-Question Robustness Analysis")
    print(f"Models: {len(models_to_evaluate)}")
    print(f"Datasets: {len(selected_datasets)}")
    print(f"Shots: {shots_to_evaluate}")
    print(f"Processes: {num_processes}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Force overwrite: {'enabled' if force_overwrite else 'disabled (skip existing)'}")
    print("Will create histogram for each model showing question-level robustness across all datasets")
    print("=" * 80)

    # Create analyzer and perform analysis
    with DataManager(use_cache=use_cache, persistent_cache_dir=get_cache_directory()) as data_manager:
        analyzer = RobustnessAnalyzer()

        # Analysis for each model separately
        total_analyses = 0

        for model in tqdm(models_to_evaluate, desc="Processing models"):
            # Early check if file already exists - before loading data
            safe_model_name = model.replace('/', '_').replace('-', '_')
            filename = f"robustness_histogram"
            model_output_dir = f'{output_dir}/{safe_model_name}'
            output_png = f'{model_output_dir}/{filename}.png'

            if not force_overwrite and os.path.exists(output_png):
                print(f"⏭️  Skipping {model} - file already exists: {output_png}")
                continue

            print(f"\n{'=' * 60}")
            print(f"Processing model: {model}")
            print(f"{'=' * 60}")

            analyzer.create_robustness_analysis(
                model_name=model,
                datasets=selected_datasets,
                shots_list=shots_to_evaluate,
                data_manager=data_manager,
                output_dir=output_dir,
                force_overwrite=force_overwrite
            )
            total_analyses += 1

        print("\n" + "=" * 80)
        print(f"✓ Per-Question Robustness Analysis completed!")
        print(f"  - Analyzed {total_analyses} models")
        print(f"  - Plots saved to: {output_dir}")
        print(f"  - Each model has its own directory with histogram")
        print("=" * 80)


if __name__ == "__main__":
    main()
