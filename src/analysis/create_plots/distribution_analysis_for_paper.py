import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.create_plots.nova_v11 import DataProcessor


@dataclass
class DistributionAnalysisConfig:
    """Configuration for distribution analysis"""
    factors: List[str] = None
    output_dir: str = '.'
    aggregation_type: Optional[str] = None
    selected_mmlu_datasets: Optional[List[str]] = None
    figsize: tuple = (15, 6)  # Reduced height as requested

    def __post_init__(self):
        if self.factors is None:
            self.factors = ["template", "separator", "enumerator", "choices_order"]


class DistributionAnalyzer:
    def __init__(self, config: DistributionAnalysisConfig):
        self.config = config
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']

        # Template mapping as provided
        self.template_mapping = {
            'MultipleChoiceTemplatesInstructionsWithTopic': 'mmlu_paper',
            'MultipleChoiceTemplatesInstructionsWithoutTopicFixed': 'mmlu_paper_without_topic',
            'MultipleChoiceTemplatesInstructionsWithTopicHelm': 'helm_with_topic',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed': 'helm_without_topic',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHarness': 'lm_evaluation_harness',
            'MultipleChoiceTemplatesStructuredWithTopic': 'structured_with_topic',
            'MultipleChoiceTemplatesStructuredWithoutTopic': 'structured_without_topic',
            'MultipleChoiceTemplatesInstructionsProSAAddress': 'paraphrase_1',
            'MultipleChoiceTemplatesInstructionsProSACould': 'paraphrase_2',
            'MultipleChoiceTemplatesInstructionsStateBelow':  'paraphrase_3',
            'MultipleChoiceTemplatesInstructionsProSASimple': 'paraphrase_4',
            'MultipleChoiceTemplatesInstructionsStateHere': 'paraphrase_5',
            'MultipleChoiceTemplatesInstructionsStateBelowPlease': 'paraphrase_6',
        }

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to avoid modifying the original
        processed_df = df.copy()

        # Shorten model names by taking the last part after splitting by '/'
        processed_df['model_name'] = processed_df['model_name'].apply(lambda x: x.split('/')[-1])

        # Replace '\n' with '\\n' in separator column
        if 'separator' in processed_df.columns:
            processed_df['separator'] = processed_df['separator'].replace('\n', '\\n')

        # Map template names if template column exists
        if 'template' in processed_df.columns:
            processed_df['template'] = processed_df['template'].map(self.template_mapping)

        return processed_df

    def _get_ordered_values(self, df: pd.DataFrame, factor: str):
        """Get values in the correct order based on factor type"""
        if factor == 'template':
            # Get unique values that exist in the data
            existing_values = set(df[factor].unique())
            # Get ordered values from mapping that exist in the data
            ordered_values = [v for v in self.template_mapping.values() if v in existing_values]
            # Add any values that might be in the data but not in mapping (shouldn't happen, but just in case)
            return ordered_values + [v for v in existing_values if v not in ordered_values]
        else:
            # For other factors, use regular sorting
            return sorted(df[factor].unique())

    def _plot_factor_distribution(self, df: pd.DataFrame, factor: str):
        """Create a vertical bar plot with adjusted spacing"""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Calculate statistics
        stats = df.groupby(['model_name', factor])['accuracy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()

        models = sorted(df['model_name'].unique())
        factor_values = self._get_ordered_values(df, factor)

        # Reduce spacing between model groups
        total_width = 0.8
        bar_width = total_width / len(factor_values)
        x = np.arange(len(models)) * (1.2)  # Reduced spacing between groups from 1.8 to 1.2

        for i, value in enumerate(factor_values):
            value_stats = stats[stats[factor] == value]
            positions = x + (i * bar_width)
            if factor == "template":
                factor_n = "Instruction"
            bars = ax.bar(positions,
                          value_stats['mean'],
                          bar_width * 0.9,
                          label=f'{factor_n}={value}',
                          color=self.colors[i % len(self.colors)],
                          alpha=0.7)

            for pos, mean in zip(positions, value_stats['mean']):
                ax.text(pos, mean + 0.001, f'{mean:.2f}',
                        ha='center', va='bottom',
                        rotation=90, fontsize=8)

        ax.set_ylabel('Accuracy')
        ax.set_title(f'Average Accuracy by {factor_n} for Each Model')
        ax.set_xticks(x + (total_width / 2) * (len(factor_values) - 1) / len(factor_values))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        legend = ax.legend(title=f'{factor_n} Values',
                           bbox_to_anchor=(1.05, 1),
                           loc='upper left',
                           ncol=max(1, len(factor_values) // 8))

        plt.tight_layout()
        output_path = f"{self.config.output_dir}/distribution_{factor}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def analyze_distributions(self, df: pd.DataFrame, metadata_path: Optional[str] = None):
        """Create distribution plots for each factor"""
        # Preprocess the data first
        processed_df = self._preprocess_data(df)

        os.makedirs(self.config.output_dir, exist_ok=True)

        stats = {}
        for factor in self.config.factors:
            if factor == "template":
                # Filter templates with less than 3000 occurrences
                processed_df = processed_df[
                    processed_df.groupby('template')['template'].transform('size') > 3000
                    ]
            self._plot_factor_distribution(processed_df, factor)
            stats[factor] = self._compute_factor_statistics(processed_df, factor)

        return stats

    def _compute_factor_statistics(self, df: pd.DataFrame, factor: str):
        """Compute summary statistics"""
        stats = df.groupby(['model_name', factor])['accuracy'].agg([
            'count',
            'mean',
            'std',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ]).round(2)

        stats = stats.rename(columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'})
        return stats


class DistributionAnalyzer1:
    """Analyzes and visualizes distributions of scores across different factors"""

    def __init__(self, config: DistributionAnalysisConfig):
        self.config = config
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']

    def analyze_distributions(self, df: pd.DataFrame, metadata_path: Optional[str] = None):
        """Create distribution plots for each factor"""
        processor = DataProcessor(df, metadata_path)

        if self.config.selected_mmlu_datasets is not None:
            processor.filter_datasets(self.config.selected_mmlu_datasets)

        if self.config.aggregation_type:
            processor.aggregate_mmlu_data(self.config.aggregation_type)

        processed_df = processor.df
        os.makedirs(self.config.output_dir, exist_ok=True)

        stats = {}
        for factor in self.config.factors:
            if factor == "template":
                # remove all the rows that here tamplates less the 2000
                processed_df = processed_df[processed_df.groupby('template')['template'].transform('size') > 3000]
            self._plot_factor_distribution(processed_df, factor)
            stats[factor] = self._compute_factor_statistics(processed_df, factor)

        return stats

    def _plot_factor_distribution(self, df: pd.DataFrame, factor: str):
        """Create a vertical bar plot with adjusted spacing for many values"""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Calculate statistics
        stats = df.groupby(['model_name', factor])['accuracy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()

        # Set up plot parameters
        models = sorted(df['model_name'].unique())
        factor_values = sorted(df[factor].unique())

        # Adjust bar width based on number of values
        total_width = 0.8  # Total width available for each model group
        bar_width = total_width / len(factor_values)

        # Create x-positions for models
        x = np.arange(len(models)) * (1 + total_width)  # Add extra space between model groups

        # Plot bars for each factor value
        for i, value in enumerate(factor_values):
            value_stats = stats[stats[factor] == value]
            positions = x + (i * bar_width)
            if factor == 'template':
                factor = "Instruction"
            bars = ax.bar(positions,
                          value_stats['mean'],
                          bar_width * 0.9,  # Slightly narrow bars to create space
                          label=f'{factor}={value}',
                          color=self.colors[i % len(self.colors)],
                          alpha=0.7)

            # Add value labels on top of bars
            for pos, mean in zip(positions, value_stats['mean']):
                ax.text(pos, mean + 0.001, f'{mean:.2f}',
                        ha='center', va='bottom',
                        rotation=90, fontsize=8)

        # Customize plot
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Average Accuracy by {factor} for Each Model')

        # Set x-ticks at center of each model group
        ax.set_xticks(x + (total_width / 2) * (len(factor_values) - 1) / len(factor_values))
        ax.set_xticklabels(models, rotation=45, ha='right')

        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Adjust legend
        legend = ax.legend(title=f'{factor} Values',
                           bbox_to_anchor=(1.05, 1),
                           loc='upper left',
                           ncol=max(1, len(factor_values) // 8))  # Split legend into columns if many values

        # Adjust layout
        plt.tight_layout()

        # Save plot
        output_path = f"{self.config.output_dir}/distribution_{factor}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def _compute_factor_statistics(self, df: pd.DataFrame, factor: str):
        """Compute summary statistics"""
        stats = df.groupby(['model_name', factor])['accuracy'].agg([
            'count',
            'mean',
            'std',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ]).round(2)

        stats = stats.rename(columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'})
        return stats

    # Run analysis


if __name__ == "__main__":
    # Load data
    df = pd.read_parquet("aggregated_results.parquet")
    # df = df[~df["dataset"].str.startswith("mmlu_pro")]
    # df = df[~df["dataset"].str.startswith("hell")]
    df = df[df["shots"] == 0]
    # use_random = False
    # if use_random:
    #     mmlu_datasets = [ds for ds in df['dataset'].unique() if ds.startswith('mmlu.')]
    #     selected_datasets = random.sample(mmlu_datasets, 5)
    #     # take only dataset that start with mmlu and in the selected_datasets ot not start with mmlu
    #     df = df[df['dataset'].isin(selected_datasets) | ~df['dataset'].str.startswith('mmlu.')]
    #
    # else:
    #     # Constants
    #     DEFAULT_DATASETS = [
    #         "mmlu.astronomy",
    #         "mmlu.college_biology",
    #         "mmlu.computer_security",
    #         "mmlu.high_school_european_history",
    #         "mmlu.high_school_government_and_politics",
    #         "ai2_arc.arc_challenge",
    #         "ai2_arc.arc_easy",
    #         "hellaswag",
    #         "openbook_qa",
    #         "social_iqa"
    #     ]
    #     df = df[df['dataset'].isin(DEFAULT_DATASETS)]

    # df = df[~df.choices_order.isin(["correct_first", "correct_last"])]

    # Configure analysis
    config = DistributionAnalysisConfig(
        factors=["template" ],
        output_dir="results",
        aggregation_type="individual",
        figsize=(15, 8)
    )
    # config = DistributionAnalysisConfig(
    #     factors=["template", "separator", "enumerator", "choices_order"],
    #     output_dir="results",
    #     aggregation_type="individual",
    #     figsize=(15, 8)
    # )

    # Run analysis
    analyzer = DistributionAnalyzer(config)
    metadata_path = "/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/mmlu_metadata.csv"

    stats = analyzer.analyze_distributions(df, metadata_path)
