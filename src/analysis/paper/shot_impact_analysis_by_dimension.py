import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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
        """Create a vertical bar plot with minimal spacing and edge alignment"""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Calculate statistics and convert to percentages
        stats = df.groupby(['model_name', factor])['accuracy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        stats['mean'] = stats['mean'] * 100  # Convert to percentage

        models = sorted(df['model_name'].unique())
        factor_values = self._get_ordered_values(df, factor)

        num_models = len(models)
        num_factors = len(factor_values)

        # הגדרת רוחב כולל
        plot_width = num_models  # נשתמש במספרים שלמים כדי לשלוט במדויק
        bar_width = 0.8 / num_factors  # צמצום הרווחים הכלליים

        # נוודא שהמודלים צמודים לשוליים באמצעות np.arange
        x = np.arange(num_models)  # זה נותן מיקומים שלמים מ-0 עד num_models-1

        for i, value in enumerate(factor_values):
            value_stats = stats[stats[factor] == value]
            positions = x + (i - (num_factors - 1) / 2) * bar_width  # מרכז כל בר בתוך הקבוצה
            factor_n = "Instruction" if factor == "template" else factor

            bars = ax.bar(positions,
                          value_stats['mean'],
                          bar_width * 0.9,
                          label=f'{factor_n}={value}',
                          color=self.colors[i % len(self.colors)],
                          alpha=0.7)

            for pos, mean in zip(positions, value_stats['mean']):
                ax.text(pos, mean + 0.1, f'{mean:.1f}',
                        ha='center', va='bottom',
                        rotation=90, fontsize=8)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Average Accuracy by {factor_n} for Each Model')

        # התאמת תוויות
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')

        # הורדת הרווחים הלבנים ע"י שליטה על הגבולות
        ax.set_xlim(-0.5, num_models - 0.5)

        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Set y-axis limits
        ax.set_ylim(20, 60)

        legend = ax.legend(title=f'{factor_n} Values',
                           bbox_to_anchor=(1.05, 1),
                           loc='upper left',
                           ncol=max(1, num_factors // 8))

        plt.tight_layout()
        output_path = f"{self.config.output_dir}/distribution_{factor}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def analyze_distributions(self, df: pd.DataFrame, metadata_path: Optional[str] = None):
        """Create distribution plots for all factors in a single figure with split bars for 0-shot and 5-shot"""
        # Preprocess the data first
        processed_df = self._preprocess_data(df)

        # Filter for shots = 0 and 5
        processed_df = processed_df[processed_df["shots"].isin([0, 5])]

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Calculate number of subplots needed
        n_factors = len(self.config.factors)

        # Create a single figure with subplots
        fig, axes = plt.subplots(1, n_factors, figsize=(self.config.figsize[0] * n_factors, self.config.figsize[1]))

        stats = {}
        for idx, factor in enumerate(self.config.factors):
            if factor == "template":
                # Filter templates with less than 3000 occurrences
                temp_df = processed_df[
                    processed_df.groupby(['template', 'shots'])['template'].transform('size') > 3000
                    ]
            else:
                temp_df = processed_df

            # Plot on the corresponding subplot
            self._plot_factor_distribution_subplot(temp_df, factor, axes[idx])
            stats[factor] = self._compute_factor_statistics(temp_df, factor)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the combined figure
        output_path = f"{self.config.output_dir}/combined_distributions_split_shots.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        return stats

    def _plot_factor_distribution_subplot(self, df: pd.DataFrame, factor: str, ax):
        """
        Creates a split bar plot showing 0-shot and 5-shot accuracy results.

        Args:
            df: DataFrame with model results
            factor: Factor to analyze (e.g. 'template')
            ax: Matplotlib axis to plot on
        """
        # Prepare data
        stats = (df.groupby(['model_name', factor, 'shots'])['accuracy']
                 .agg(['mean', 'std', 'count'])
                 .reset_index())
        stats['mean'] *= 100  # Convert to percentage

        models = sorted(df['model_name'].unique())
        factor_values = self._get_ordered_values(df, factor)

        # Set up plot dimensions with increased spacing
        bar_width = 0.6 / len(factor_values)  # Increased from 0.4
        x_positions = np.linspace(0, 1.5, len(models), endpoint=True)  # Increased range from 1.0 to 1.5

        # Plot bars for each factor value
        for i, value in enumerate(factor_values):
            value_data = stats[stats[factor] == value]
            shot_0_data = value_data[value_data['shots'] == 0]
            shot_5_data = value_data[value_data['shots'] == 5]

            # Calculate bar positions
            offset = (i - (len(factor_values) - 1) / 2) * bar_width
            bar_positions = x_positions + offset

            # Set colors
            base_color = plt.matplotlib.colors.to_rgb(self.colors[i % len(self.colors)])
            color_5_shot = base_color
            color_0_shot = tuple(0.4 + 0.6 * c for c in base_color)

            # Plot 0-shot bars
            factor_label = "Instruction" if factor == "template" else factor
            ax.bar(bar_positions,
                   shot_0_data['mean'],
                   bar_width * 0.9,
                   label=f'{factor_label}={value} (0-shot)',
                   color=color_0_shot,
                   alpha=0.9)

            # Plot 5-shot bars if data exists
            if not shot_5_data['mean'].empty:
                height_5_shot = shot_5_data['mean'].values - shot_0_data['mean'].values
                ax.bar(bar_positions,
                       height_5_shot,
                       bar_width * 0.9,
                       bottom=shot_0_data['mean'],
                       label=f'{factor_label}={value} (5-shot)',
                       color=color_5_shot,
                       alpha=0.9)

                # Add value labels
                self._add_value_labels(ax, bar_positions,
                                       shot_0_data['mean'],
                                       shot_5_data['mean'])

        self._setup_axes(ax, models, factor)

    def _add_value_labels(self, ax, positions, values_0, values_5):
        """Add text labels to the bars."""
        for pos, val_0, val_5 in zip(positions, values_0, values_5):
            ax.text(pos, val_0 / 1.2, f'{val_0:.1f}',
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
            ax.text(pos, (val_0 + val_5) / 2, f'{val_5:.1f}',
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    def _setup_axes(self, ax, models, factor):
        """Configure axis properties."""
        ax.set_xticks(np.linspace(0, 1.5, len(models), endpoint=True))  # Increased range
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_xlim(-0.3, 1.8)  # Adjusted limits to match increased spacing
        ax.margins(x=0.1)  # Increased margins
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(20, 80)

        # Only set ylabel for leftmost subplot
        if ax.get_position().x0 < 0.1:
            ax.set_ylabel('Accuracy (%)')

        factor_name = "Instruction" if factor == "template" else factor
        ax.set_title(f'Average Accuracy by {factor_name}\n(Split bars: bottom=0-shot, top=5-shot)')
    def _plot_factor_distribution_subplot_work(self, df: pd.DataFrame, factor: str, ax):
        """Create a vertical split bar plot on the given subplot showing 0-shot and 5-shot results"""
        # Calculate statistics for both 0-shot and 5-shot, convert to percentages
        stats = df.groupby(['model_name', factor, 'shots'])['accuracy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        stats['mean'] = stats['mean'] * 100  # Convert to percentage

        models = sorted(df['model_name'].unique())
        factor_values = self._get_ordered_values(df, factor)

        num_models = len(models)
        num_factors = len(factor_values)

        # Define spacing parameters
        plot_width = num_models * 0.1  # הקטנה מ-0.2 ל-0.1 לצמצום המרחק בין מודלים
        bar_width = 0.4 / num_factors  # הקטנה מ-0.6 ל-0.4 לעמודות דקות יותר
        plot_width = 1.0
        x = np.linspace(0, plot_width, num_models, endpoint=True)

        for i, value in enumerate(factor_values):
            value_stats = stats[stats[factor] == value]

            # Separate 0-shot and 5-shot data
            shot_0_stats = value_stats[value_stats['shots'] == 0]
            shot_5_stats = value_stats[value_stats['shots'] == 5]

            positions = x + (i - (num_factors - 1) / 2) * bar_width
            factor_n = "Instruction" if factor == "template" else factor

            # Get base color and convert to RGB
            base_color = plt.matplotlib.colors.to_rgb(self.colors[i % len(self.colors)])

            # Create two different shades
            color_5 = base_color
            color_0 = tuple(0.4 + 0.6 * c for c in base_color)  # Lighter shade for 5-shot

            # Plot bars for 0-shot (bottom)
            bars_0 = ax.bar(positions,
                            shot_0_stats['mean'],
                            bar_width * 0.9,
                            label=f'{factor_n}={value} (0-shot)',
                            color=color_0,
                            alpha=0.9)

            if shot_5_stats['mean'].values.size == 0:
                continue

            # Plot bars for 5-shot (top)
            bars_5 = ax.bar(positions,
                            shot_5_stats['mean'].values - shot_0_stats['mean'].values,
                            bar_width * 0.9,
                            bottom=shot_0_stats['mean'],
                            label=f'{factor_n}={value} (5-shot)',
                            color=color_5,
                            alpha=0.9)

            for pos, mean_0, mean_5 in zip(positions, shot_0_stats['mean'], shot_5_stats['mean']):
                ax.text(pos, mean_0 / 1.2, f'{mean_0:.1f}',
                        ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
                ax.text(pos, (mean_0 + mean_5) / 2, f'{mean_5:.1f}',
                        ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')

        # Adjust labels
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_xlim(-0.22, 1.22)
        ax.margins(x=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Set y-axis limits
        ax.set_ylim(20, 80)

        # Only set ylabel for the leftmost subplot
        if ax.get_position().x0 < 0.1:
            ax.set_ylabel('Accuracy (%)')

        ax.set_title(f'Average Accuracy by {factor_n}\n(Split bars: bottom=0-shot, top=5-shot)')


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



if __name__ == "__main__":
    # Load data
    df = pd.read_parquet("aggregated_results.parquet")
    # df = df[~df["dataset"].str.startswith("mmlu_pro")]
    # df = df[~df["dataset"].str.startswith("hell")]
    models = [
        # 'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        # 'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3'
    ]
    # df = df[df["shots"] == 0]
    df = df[df["model_name"].isin(models)]
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

    df = df[
        ((df['shots'] != 5) | (~df['choices_order'].isin(["correct_first", "correct_last"])))
    ]
    # Configure analysis
    config = DistributionAnalysisConfig(
        factors=[ "enumerator", "template", "separator"],
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
    # models = [
    #     'meta-llama/Llama-3.2-1B-Instruct',
    #     'allenai/OLMoE-1B-7B-0924-Instruct',
    #     'meta-llama/Meta-Llama-3-8B-Instruct',
    #     'meta-llama/Llama-3.2-3B-Instruct',
    #     'mistralai/Mistral-7B-Instruct-v0.3'
    # ]
    # Run analysis
    analyzer = DistributionAnalyzer(config)
    metadata_path = "/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/mmlu_metadata.csv"

    stats = analyzer.analyze_distributions(df, metadata_path)
