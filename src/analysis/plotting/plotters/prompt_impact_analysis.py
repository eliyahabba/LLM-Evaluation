#!/usr/bin/env python3
"""
Improved Prompt Elements Impact Analyzer

Analyzes the impact of different prompt elements on model performance.
This script creates detailed plots showing how various prompt components 
(templates, enumerators, separators, choice ordering) affect model accuracy.
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend - no display windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.analysis.plotting.utils.config import (
    DEFAULT_MODELS, DEFAULT_DATASETS, DEFAULT_SHOTS, DEFAULT_NUM_PROCESSES,
    PLOT_STYLE,
    get_model_display_name, format_dataset_name
)
from src.analysis.plotting.utils.data_manager import DataManager
from src.analysis.plotting.utils.auth import ensure_hf_authentication
from src.analysis.plotting.utils.config import get_cache_directory


class PromptImpactAnalyzer:
    """
    Analyzes the impact of different prompt elements on model performance.
    
    This class creates visualization plots showing how various prompt components
    affect model accuracy across different datasets. It generates separate graphs
    for each dataset and model combination.
    """

    def __init__(self):
        """Initialize the analyzer with consistent plot styling and element mappings."""
        # Style settings from config file
        plt.rcParams['font.family'] = PLOT_STYLE['font_family']
        plt.rcParams['font.serif'] = PLOT_STYLE['font_serif']
        plt.rcParams['figure.dpi'] = PLOT_STYLE['figure_dpi']

        # Colors for different components
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']

        # Template name mapping (from original file)
        self.template_mapping = {
            # From BasicMCPrompts.py
            'MultipleChoiceTemplatesInstructionsWithTopic': 'mmlu_paper',
            'MultipleChoiceTemplatesInstructionsWithoutTopic': 'mmlu_paper_without_topic_original',
            'MultipleChoiceTemplatesInstructionsWithoutTopicFixed': 'mmlu_paper_without_topic',
            'MultipleChoiceTemplatesInstructionsWithTopicHelm': 'helm_with_topic',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHelm': 'helm_without_topic_original',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed': 'helm_without_topic',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHarness': 'lm_evaluation_harness',
            'MultipleChoiceTemplatesStructuredWithTopic': 'structured_with_topic',
            'MultipleChoiceTemplatesStructuredWithoutTopic': 'structured_without_topic',
            'MultipleChoiceTemplatesInstructionsProSAAddress': 'paraphrase_1',
            'MultipleChoiceTemplatesInstructionsProSACould': 'paraphrase_2',
            'MultipleChoiceTemplatesInstructionsStateBelow': 'paraphrase_3',
            'MultipleChoiceTemplatesInstructionsProSASimple': 'paraphrase_4',
            'MultipleChoiceTemplatesInstructionsStateHere': 'paraphrase_5',
            'MultipleChoiceTemplatesInstructionsStateBelowPlease': 'paraphrase_6',

            # From SocialQaPrompts.py and OpenBookQAPrompts.py
            'MultipleChoiceTemplatesInstructionsWithTopicAndCoT': 'chain_of_thought',

            # From HellaSwagPrompts.py
            'MultipleChoiceTemplatesInstructionsStandard': 'hellaswag_standard',
            'MultipleChoiceTemplatesInstructionsContext': 'hellaswag_context',
            'MultipleChoiceTemplatesInstructionsStructured': 'hellaswag_structured',
            'MultipleChoiceTemplatesInstructionsBasic': 'hellaswag_basic',
            'MultipleChoiceTemplatesInstructionsState1': 'hellaswag_state_1',
            'MultipleChoiceTemplatesInstructionsState2': 'hellaswag_state_2',
            'MultipleChoiceTemplatesInstructionsState3': 'hellaswag_state_3',
            'MultipleChoiceTemplatesInstructionsState4': 'hellaswag_state_4',
            'MultipleChoiceTemplatesInstructionsState5': 'hellaswag_state_5',
            'MultipleChoiceTemplatesInstructionsState6': 'hellaswag_state_6',
            'MultipleChoiceTemplatesInstructionsState7': 'hellaswag_state_7',
            'MultipleChoiceTemplatesInstructionsState8': 'hellaswag_state_8',

            # From RacePrompts.py
            'MultipleChoiceContextTemplateBasic': 'race_basic',
            'MultipleChoiceContextTemplateBasicNoContextLabel': 'race_basic_no_label',
            'MultipleChoiceContextTemplateMMluStyle': 'race_mmlu_style',
            'MultipleChoiceContextTemplateMMluHelmStyle': 'race_mmlu_helm_style',
            'MultipleChoiceContextTemplateMMluHelmWithChoices': 'race_mmlu_helm_choices',
            'MultipleChoiceContextTemplateProSASimple': 'race_prosa_simple',
            'MultipleChoiceContextTemplateProSACould': 'race_prosa_could',
            'MultipleChoiceContextTemplateStateNumbered': 'race_state_numbered',
            'MultipleChoiceContextTemplateStateOptions': 'race_state_options',
            'MultipleChoiceContextTemplateStateSelect': 'race_state_select',
            'MultipleChoiceContextTemplateStateRead': 'race_state_read',
            'MultipleChoiceContextTemplateStateMultipleChoice': 'race_state_multiple_choice',
        }

        # Enumerator mappings
        self.enumerator_mapping = {
            "capitals": "A,B,C,D",
            "lowercase": "a,b,c,d",
            "numbers": "1,2,3,4",
            "roman": "I,II,III,IV",
            "keyboard": "!,@,#,$",
            "greek": "α,β,γ,δ"
        }

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data to adapt to expected format.
        
        Args:
            df: Raw DataFrame from data manager
            
        Returns:
            Processed DataFrame with renamed columns and mapped values
        """
        processed_df = df.copy()

        # Change column names to format expected by original analyzer
        column_mapping = {
            'dimensions_4: instruction_phrasing_text': 'template',
            'dimensions_1: enumerator': 'enumerator',
            'dimensions_2: separator': 'separator',
            'dimensions_3: choices_order': 'choices_order'
        }

        for old_col, new_col in column_mapping.items():
            processed_df[new_col] = processed_df[old_col]

        # Shorten model names
        processed_df['model_name'] = processed_df['model'].apply(lambda x: x.split('/')[-1])
        processed_df['model_name'] = processed_df['model_name'].apply(lambda x: x.replace('Meta-', ''))

        # Fix separators
        processed_df['separator'] = processed_df['separator'].str.replace('\n', '\\n')

        # Map templates
        processed_df['template'] = processed_df['template'].map(self.template_mapping).fillna(
            processed_df['template'])

        # Map enumerators
        processed_df['enumerator'] = processed_df['enumerator'].map(self.enumerator_mapping).fillna(
            processed_df['enumerator'])

        return processed_df

    def create_analysis_plots(
            self,
            data: pd.DataFrame,
            dataset_name: str,
            models: List[str],
            factors: List[str] = None,
            shots: int = 0,
            output_dir: str = "plots",
            force_overwrite: bool = False
    ):
        """
        Create prompt elements analysis for one dataset - separate graph for each model.
        
        Args:
            data: DataFrame containing evaluation results
            dataset_name: Name of the dataset to analyze
            models: List of model names to include
            factors: List of prompt factors to analyze
            shots: Number of shots (0 for zero-shot, 5 for few-shot)
            output_dir: Directory to save plots
            force_overwrite: Whether to overwrite existing files
        """
        if factors is None:
            factors = ["template", "enumerator", "separator", "choices_order"]

        # Filter data for specific dataset and shots
        dataset_data = data[
            (data['dataset'] == dataset_name) &
            (data['dimensions_5: shots'] == shots)
            ].copy()

        if dataset_data.empty:
            print(f"⚠️  No data found for dataset: {dataset_name}, shots: {shots} - skipping")
            return

        # Preprocessing
        processed_data = self._preprocess_data(dataset_data)

        # Check which models have data
        models_with_data = []
        for model in models:
            model_short = model.split('/')[-1].replace('Meta-', '')
            if model_short in processed_data['model_name'].values:
                models_with_data.append(model)
            else:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")

        if not models_with_data:
            print(f"⚠️  No models have data for dataset {dataset_name}, shots: {shots} - skipping plot")
            return

        # Filter factors that exist in data
        available_factors = [f for f in factors if not processed_data[f].isna().all()]

        if not available_factors:
            print(f"⚠️  No valid factors found for dataset {dataset_name} - skipping")
            return

        print(
            f"📊 Creating prompt elements analysis for {dataset_name} with {len(models_with_data)} models and {len(available_factors)} factors")
        print(f"   Will create separate plot for each model")

        shots_text = "Zero-shot" if shots == 0 else f"{shots}-shot"
        formatted_dataset_name = format_dataset_name(dataset_name)

        # Create separate graph for each model
        for model in models_with_data:
            # Check if file already exists for this model
            model_short_name = model.split('/')[-1]
            model_output_dir = f'{output_dir}/{model_short_name}'
            safe_dataset_name = dataset_name.replace('.', '_').replace('/', '_')
            filename = f"accuracy_marginalization_{shots}shot"

            # Create model and dataset directory
            dataset_output_dir = f'{model_output_dir}/{safe_dataset_name}'
            Path(dataset_output_dir).mkdir(parents=True, exist_ok=True)

            output_png = f'{dataset_output_dir}/{filename}.png'

            if not force_overwrite and os.path.exists(output_png):
                print(f"⏭️  Skipping {model_short_name}/{dataset_name} ({shots}shot) - file already exists")
                continue

            # Filter data for this specific model
            model_short = model.split('/')[-1].replace('Meta-', '')
            model_data = processed_data[processed_data['model_name'] == model_short]

            if model_data.empty:
                print(f"⚠️  No processed data for model {model_short} - skipping")
                continue

            # Set up the plot
            n_factors = len(available_factors)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            # White background
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)

            for i, factor in enumerate(available_factors):
                ax = axes[i]
                ax.set_facecolor('white')

                # Group by factor and calculate mean and std
                factor_groups = model_data.groupby(factor)['accuracy'].agg(['mean', 'std', 'count']).reset_index()
                factor_groups = factor_groups.sort_values('mean', ascending=False)

                # Prepare data for plotting
                factor_values = factor_groups[factor].values
                means = factor_groups['mean'].values * 100  # Convert to percentage
                stds = factor_groups['std'].values * 100
                counts = factor_groups['count'].values

                # Replace NaN std with 0
                stds = np.nan_to_num(stds)

                # Create bar plot
                bars = ax.bar(range(len(factor_values)), means,
                              yerr=stds, capsize=5, alpha=0.7,
                              color=self.colors[i % len(self.colors)])

                # Add value labels on bars
                for j, (bar, mean_val, count) in enumerate(zip(bars, means, counts)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + stds[j] + 0.5,
                            f'{mean_val:.1f}%\n(n={count})',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

                # Customize the subplot
                ax.set_xlabel(factor.replace('_', ' ').title(), fontsize=14, fontweight='bold')
                ax.set_ylabel('Accuracy (%)', fontsize=12)
                ax.set_title(f'{factor.replace("_", " ").title()} Impact', fontsize=16, fontweight='bold')

                # Set x-axis labels
                ax.set_xticks(range(len(factor_values)))
                ax.set_xticklabels([str(v) for v in factor_values], rotation=45, ha='right')

                # Set y-axis limits
                ax.set_ylim(0, min(100, max(means + stds) * 1.1))

                # Add grid
                ax.grid(True, alpha=0.3, axis='y')

            # Hide unused subplots
            for i in range(n_factors, 4):
                axes[i].set_visible(False)

            # Main title
            main_title = f'{formatted_dataset_name} - {get_model_display_name(model)} ({shots_text})'
            fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.98)

            plt.tight_layout()

            # Save the plot
            plt.savefig(output_png,
                        dpi=PLOT_STYLE['figure_dpi'],
                        bbox_inches=PLOT_STYLE['bbox_inches'],
                        transparent=PLOT_STYLE['transparent'],
                        facecolor=PLOT_STYLE['facecolor'])
            plt.savefig(f'{dataset_output_dir}/{filename}.svg',
                        bbox_inches=PLOT_STYLE['bbox_inches'],
                        transparent=PLOT_STYLE['transparent'],
                        facecolor=PLOT_STYLE['facecolor'])

            plt.close()
            print(f"✅ Created prompt elements plot for {model_short_name}/{dataset_name} ({shots}shot)")

    def create_combined_analysis(
            self,
            data: pd.DataFrame,
            dataset_name: str,
            models: List[str],
            factors: List[str] = None,
            output_dir: str = "plots",
            force_overwrite: bool = False
    ):
        """
        Create combined prompt elements analysis (0-shot and 5-shot together) for one dataset.
        
        This creates a comprehensive analysis that includes shots as an additional dimension,
        showing how prompt elements interact with few-shot learning.
        
        Args:
            data: DataFrame containing evaluation results
            dataset_name: Name of the dataset to analyze
            models: List of model names to include
            factors: List of prompt factors to analyze (defaults to 5 including shots)
            output_dir: Directory to save plots
            force_overwrite: Whether to overwrite existing files
        """
        if factors is None:
            factors = ["template", "enumerator", "separator", "choices_order", "shots"]

        # Filter data for specific dataset (all shots)
        dataset_data = data[data['dataset'] == dataset_name].copy()

        if dataset_data.empty:
            print(f"⚠️  No data found for dataset: {dataset_name} - skipping combined analysis")
            return

        # Preprocessing
        processed_data = self._preprocess_data(dataset_data)

        # Add shots as a factor
        processed_data['shots'] = processed_data['dimensions_5: shots'].astype(str) + '-shot'

        # Check which models have data
        models_with_data = []
        for model in models:
            model_short = model.split('/')[-1].replace('Meta-', '')
            if model_short in processed_data['model_name'].values:
                models_with_data.append(model)
            else:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")

        if not models_with_data:
            print(f"⚠️  No models have data for dataset {dataset_name} - skipping combined plot")
            return

        # Filter factors that exist in data
        available_factors = [f for f in factors if not processed_data[f].isna().all()]

        if not available_factors:
            print(f"⚠️  No valid factors found for dataset {dataset_name} - skipping combined analysis")
            return

        print(f"📊 Creating COMBINED prompt elements analysis for {dataset_name} with {len(models_with_data)} models")
        print(f"   Factors: {available_factors}")

        formatted_dataset_name = format_dataset_name(dataset_name)

        # Create separate combined graph for each model
        for model in models_with_data:
            model_short_name = model.split('/')[-1]
            model_output_dir = f'{output_dir}/{model_short_name}'
            safe_dataset_name = dataset_name.replace('.', '_').replace('/', '_')
            filename = f"accuracy_marginalization_combined"

            # Create model and dataset directory
            dataset_output_dir = f'{model_output_dir}/{safe_dataset_name}'
            Path(dataset_output_dir).mkdir(parents=True, exist_ok=True)

            output_png = f'{dataset_output_dir}/{filename}.png'

            if not force_overwrite and os.path.exists(output_png):
                print(f"⏭️  Skipping {model_short_name}/{dataset_name} (combined) - file already exists")
                continue

            # Filter data for this specific model  
            model_short = model.split('/')[-1].replace('Meta-', '')
            model_data = processed_data[processed_data['model_name'] == model_short]

            if model_data.empty:
                print(f"⚠️  No processed data for model {model_short} - skipping combined")
                continue

            # Determine subplot layout based on number of factors
            n_factors = len(available_factors)
            if n_factors <= 4:
                nrows, ncols = 2, 2
            elif n_factors <= 6:
                nrows, ncols = 2, 3
            else:
                nrows, ncols = 3, 3

            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
            if n_factors == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # White background
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)

            for i, factor in enumerate(available_factors):
                ax = axes[i]
                ax.set_facecolor('white')

                # Group by factor and calculate mean and std
                factor_groups = model_data.groupby(factor)['accuracy'].agg(['mean', 'std', 'count']).reset_index()
                factor_groups = factor_groups.sort_values('mean', ascending=False)

                # Prepare data for plotting
                factor_values = factor_groups[factor].values
                means = factor_groups['mean'].values * 100  # Convert to percentage
                stds = factor_groups['std'].values * 100
                counts = factor_groups['count'].values

                # Replace NaN std with 0
                stds = np.nan_to_num(stds)

                # Create bar plot
                bars = ax.bar(range(len(factor_values)), means,
                              yerr=stds, capsize=5, alpha=0.7,
                              color=self.colors[i % len(self.colors)])

                # Add value labels on bars
                for j, (bar, mean_val, count) in enumerate(zip(bars, means, counts)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + stds[j] + 0.5,
                            f'{mean_val:.1f}%\n(n={count})',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

                # Customize the subplot
                ax.set_xlabel(factor.replace('_', ' ').title(), fontsize=14, fontweight='bold')
                ax.set_ylabel('Accuracy (%)', fontsize=12)
                ax.set_title(f'{factor.replace("_", " ").title()} Impact', fontsize=16, fontweight='bold')

                # Set x-axis labels
                ax.set_xticks(range(len(factor_values)))
                ax.set_xticklabels([str(v) for v in factor_values], rotation=45, ha='right')

                # Set y-axis limits
                ax.set_ylim(0, min(100, max(means + stds) * 1.1))

                # Add grid
                ax.grid(True, alpha=0.3, axis='y')

            # Hide unused subplots
            for i in range(n_factors, nrows * ncols):
                if i < len(axes):
                    axes[i].set_visible(False)

            # Main title
            main_title = f'{formatted_dataset_name} - {get_model_display_name(model)} (Combined Analysis)'
            fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.98)

            plt.tight_layout()

            # Save the plot
            plt.savefig(output_png,
                        dpi=PLOT_STYLE['figure_dpi'],
                        bbox_inches=PLOT_STYLE['bbox_inches'],
                        transparent=PLOT_STYLE['transparent'],
                        facecolor=PLOT_STYLE['facecolor'])
            plt.savefig(f'{dataset_output_dir}/{filename}.svg',
                        bbox_inches=PLOT_STYLE['bbox_inches'],
                        transparent=PLOT_STYLE['transparent'],
                        facecolor=PLOT_STYLE['facecolor'])

            plt.close()
            print(f"✅ Created COMBINED prompt elements plot for {model_short_name}/{dataset_name}")


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description='Create prompt elements impact analysis plots for LLM evaluation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_prompt_impact_analysis.py
  python run_prompt_impact_analysis.py --models meta-llama/Llama-3.2-1B-Instruct
  python run_prompt_impact_analysis.py --datasets mmlu.college_biology
        """
    )

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'List of models to analyze (default: {len(DEFAULT_MODELS)} models)')

    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help=f'List of datasets to analyze (default: {len(DEFAULT_DATASETS)} datasets)')

    parser.add_argument('--factors', nargs='+',
                        default=['template', 'enumerator', 'separator', 'choices_order'],
                        choices=['template', 'enumerator', 'separator', 'choices_order'],
                        help='List of prompt factors to analyze (default: template enumerator separator choices_order)')

    parser.add_argument('--shots', nargs='+', type=int, default=DEFAULT_SHOTS,
                        help=f'List of shot counts to analyze (default: {DEFAULT_SHOTS})')

    parser.add_argument('--num-processes', type=int, default=DEFAULT_NUM_PROCESSES,
                        help=f'Number of parallel processes (default: {DEFAULT_NUM_PROCESSES})')

    parser.add_argument('--output-dir', default="plots/accuracy_marginalization",
                        help='Output directory for plots (default: plots/accuracy_marginalization)')

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
    Main function - processes each dataset separately for memory efficiency.
    
    Creates detailed prompt elements analysis showing how different prompt components
    (templates, enumerators, separators, choice ordering) affect model performance.
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
    factors_to_analyze = args.factors
    shots_to_evaluate = args.shots
    num_processes = args.num_processes
    use_cache = not args.no_cache
    output_dir = args.output_dir
    force_overwrite = args.force

    print("Starting Prompt Elements Impact Analysis")
    print(f"Models: {len(models_to_evaluate)}")
    print(f"Datasets: {len(selected_datasets)}")
    print(f"Factors: {factors_to_analyze}")
    print(f"Shots: {shots_to_evaluate}")
    print(f"Processes: {num_processes}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Force overwrite: {'enabled' if force_overwrite else 'disabled (skip existing)'}")
    print("MEMORY EFFICIENT: Processing one dataset at a time")
    print("=" * 60)

    # Create analyzer
    with DataManager(use_cache=use_cache, persistent_cache_dir=get_cache_directory()) as data_manager:
        analyzer = PromptImpactAnalyzer()

        # Process dataset by dataset - memory efficient
        total_plots = 0

        for dataset in tqdm(selected_datasets, desc="Processing datasets"):
            # Check which models are missing plots for this dataset
            safe_dataset_name = dataset.replace('.', '_').replace('/', '_')

            # Identify models that need plots generated (missing files)
            models_needing_plots = []
            if not force_overwrite:
                for model in models_to_evaluate:
                    model_short_name = model.split('/')[-1]
                    model_dataset_dir = f'{output_dir}/{model_short_name}/{safe_dataset_name}'

                    # Check if any plot files are missing for this model
                    model_needs_plots = False

                    # Check individual shots plots
                    for shots in shots_to_evaluate:
                        filename = f"accuracy_marginalization_{shots}shot.png"
                        output_file = f'{model_dataset_dir}/{filename}'
                        if not os.path.exists(output_file):
                            model_needs_plots = True
                            break

                    # Check combined plot if individual shots plots exist
                    if not model_needs_plots:
                        combined_filename = f"accuracy_marginalization_combined.png"
                        combined_output_file = f'{model_dataset_dir}/{combined_filename}'
                        if not os.path.exists(combined_output_file):
                            model_needs_plots = True

                    if model_needs_plots:
                        models_needing_plots.append(model)
            else:
                # If force overwrite, all models need plots
                models_needing_plots = models_to_evaluate

            # Skip if no models need plots
            if not models_needing_plots:
                print(f"⏭️  Skipping {dataset} - all plots already exist for all models")
                continue

            print(f"Loading data for dataset: {dataset}")
            print(f"  Models needing plots: {len(models_needing_plots)}/{len(models_to_evaluate)}")
            for model in models_needing_plots:
                print(f"    - {model.split('/')[-1]}")

            # Load data ONLY for models that need plots - OPTIMIZED!
            dataset_data = data_manager.load_multiple_models(
                model_names=models_needing_plots,  # Only load for models that need plots!
                datasets=[dataset],  # Only one dataset!
                shots_list=shots_to_evaluate,
                aggregate=True,
                num_processes=num_processes
            )

            if dataset_data.empty:
                print(f"⚠️  No data loaded for dataset: {dataset} - skipping")
                continue

            print(f"✓ Loaded data for {dataset}: {dataset_data.shape}")

            # Create separate graphs for each shots setting
            for shots in shots_to_evaluate:
                shots_text = "zero-shot" if shots == 0 else f"{shots}-shot"
                print(f"Creating {shots_text} prompt elements analysis for {dataset}...")

                analyzer.create_analysis_plots(
                    data=dataset_data,
                    dataset_name=dataset,
                    models=models_needing_plots,
                    factors=factors_to_analyze,
                    shots=shots,
                    output_dir=output_dir,
                    force_overwrite=force_overwrite
                )
                total_plots += 1

            # Create combined graphs (0+5 shots) with shots as additional dimension
            print(f"Creating COMBINED prompt elements analysis for {dataset}...")
            analyzer.create_combined_analysis(
                data=dataset_data,
                dataset_name=dataset,
                models=models_needing_plots,
                factors=None,  # Will use default of 5 dimensions
                output_dir=output_dir,
                force_overwrite=force_overwrite
            )
            total_plots += 1

            # Free memory
            del dataset_data
            print(f"✅ Completed {dataset} and freed memory")

        print("=" * 60)
        print(f"✓ Prompt Elements Impact Analysis completed!")
        print(f"  - Processed {total_plots} dataset-shot combinations")
        print(f"  - Each dataset processed individually (memory efficient)")
        print(f"  - Created separate plot for each model")
        print(f"  - Each plot shows all factors for that specific model")
        print(f"  - Analyzed factors: {factors_to_analyze}")
        print(f"  - For each dataset: 3 files per model:")
        print(f"    • 0-shot analysis (4 factors): template, enumerator, separator, choices_order")
        print(f"    • 5-shot analysis (4 factors): template, enumerator, separator, choices_order")
        print(f"    • Combined analysis (5 factors): template, enumerator, separator, choices_order, shots")
        print(f"  - Plots saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
