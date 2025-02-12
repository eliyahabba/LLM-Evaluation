import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

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
            'MultipleChoiceTemplatesInstructionsStateBelow': 'paraphrase_3',
            'MultipleChoiceTemplatesInstructionsProSASimple': 'paraphrase_4',
            'MultipleChoiceTemplatesInstructionsStateHere': 'paraphrase_5',
            'MultipleChoiceTemplatesInstructionsStateBelowPlease': 'paraphrase_6',
        }
        self.enumerator_mapping = {
    "capitals": "A,B,C,D",
    "lowercase": "a,b,c,d",
    "numbers": "1,2,3,4",
    "roman": "I,II,III,IV",
    "keyboard": "!,@,#,$",
    "greek": "α,β,γ,δ"
}


    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to avoid modifying the original
        processed_df = df.copy()

        # Shorten model names by taking the last part after splitting by '/'
        processed_df['model_name'] = processed_df['model_name'].apply(lambda x: x.split('/')[-1])
        # rename         ' Meta-Llama-3-8B-Instruct', Llama-3-8B-Instruct
        processed_df['model_name'] = processed_df['model_name'].apply(lambda x: x.replace('Meta-', ''))

        # Replace '\n' with '\\n' in separator column
        if 'separator' in processed_df.columns:
            processed_df['separator'] = processed_df['separator'].replace('\n', '\\n')

        # Map template names if template column exists
        if 'template' in processed_df.columns:
            processed_df['template'] = processed_df['template'].map(self.template_mapping)

        if 'enumerator' in processed_df.columns:
            processed_df['enumerator'] = processed_df['enumerator'].map(self.enumerator_mapping)

        return processed_df

    def _get_ordered_values(self, df: pd.DataFrame, factor: str):
        """Get values in the correct order based on factor type"""
        if factor == 'template':
            existing_values = set(df[factor].unique())
            ordered_values = [v for v in self.template_mapping.values() if v in existing_values]
            return ordered_values + [v for v in existing_values if v not in ordered_values]
        else:
            return sorted(df[factor].unique())

    def _plot_all_factors(self, df: pd.DataFrame):
        """Create a single figure with subplots for all factors"""
        num_factors = len(self.config.factors)
        fig, axes = plt.subplots(1, num_factors , figsize=((num_factors-1) * 8, 8))
        # fig, axes = plt.subplots(1, num_factors, figsize=(num_factors * 8, 8*(num_factors-1)))
        for idx, (factor, ax) in enumerate(zip(self.config.factors, axes)):
            if factor == "template":
                factor_df = df[df.groupby('template')['template'].transform('size') > 3000]
            else:
                factor_df = df

            # Calculate statistics
            stats = factor_df.groupby(['model_name', factor])['accuracy'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            stats['mean'] = stats['mean'] * 100  # Convert to percentage

            models = sorted(factor_df['model_name'].unique())
            factor_values = self._get_ordered_values(factor_df, factor)

            num_models = len(models)
            num_values = len(factor_values)
            bar_width = 0.8 / num_values
            x = np.arange(num_models)
            unique_values = stats[factor].unique()

            # צור מילון שממפה כל ערך לצבע קבוע
            color_map = {value: self.colors[i % len(self.colors)] for i, value in enumerate(unique_values)}

            # עבור כל מודל, נמיין את הערכים לפי הממוצע מהגבוה לנמוך
            for model_idx, model in enumerate(models):
                model_stats = stats[stats['model_name'] == model].sort_values('mean', ascending=False)
                positions = x[model_idx] + (np.arange(len(model_stats)) - (num_values - 1) / 2) * bar_width

                bars = ax.bar(positions,
                              model_stats['mean'],
                              bar_width * 0.9,
                              label=model_stats[factor].tolist(),
                              color=[color_map[val] for val in model_stats[factor]],  # שימוש בצבעים הקבועים
                              alpha=0.7)

                # הצג ערכים מספריים רק עבור הערך הגבוה והנמוך ביותר
                ax.text(positions[0], model_stats['mean'].iloc[0] + 0.1,
                        f'{model_stats["mean"].iloc[0]:.1f}',
                        # make it bold
                        ha='center', va='bottom', rotation=90, fontsize=15, fontweight='bold')

                ax.text(positions[-1], model_stats['mean'].iloc[-1] + 0.1,
                        f'{model_stats["mean"].iloc[-1]:.1f}',
                        ha='center', va='bottom', rotation=90, fontsize=15, fontweight='bold')

            # Customize each subplot
            factor_n = {
                "template": "Phrasing",
                "enumerator": "Enumerator",
                "separator": "Choice Separator"
            }.get(factor, factor)

            if idx == 0:
                ax.set_ylabel('Accuracy (%)', fontsize=20)
            else:
                ax.set_ylabel('')
                ax.yaxis.set_ticklabels([])

            ax.set_title(f'{factor_n}', fontsize=20)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='center', fontsize=18)
            ax.set_ylim(30, 60)
            # increase size of font of y sticks
            ax.yaxis.set_tick_params(labelsize=16)
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

        plt.ylim(30, 60)
        plt.tight_layout()
        output_path = f"{self.config.output_dir}/distribution_all_factors.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    def analyze_distributions(self, df: pd.DataFrame, metadata_path: Optional[str] = None):
        """Create distribution plots for all factors in a single figure"""
        processed_df = self._preprocess_data(df)
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Create single figure with all factors
        self._plot_all_factors(processed_df)

        # Compute statistics
        stats = {}
        for factor in self.config.factors:
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


if __name__ == "__main__":
    # Load data
    df = pd.read_parquet("aggregated_results.parquet")
    models = [
        # 'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        # 'meta-llama/Meta-Llama-3-8B-Instruct',
        # 'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3'
    ]
    df = df[df["model_name"].isin(models)]

    df = df[
        ((df['shots'] != 5) | (~df['choices_order'].isin(["correct_first", "correct_last"])))
    ]
    # Co
    # Co
    # Configure analysis
    config = DistributionAnalysisConfig(
        factors=["template","enumerator", "separator"],
        output_dir="results",
        aggregation_type="individual",
        figsize=(15, 8)
    )

    # Run analysis
    analyzer = DistributionAnalyzer(config)
    metadata_path = "/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/mmlu_metadata.csv"

    stats = analyzer.analyze_distributions(df, metadata_path)
