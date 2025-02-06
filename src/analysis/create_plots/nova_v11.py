import os
import random
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from tqdm import tqdm

@dataclass
class AnalysisConfig:
    """Configuration class for analysis parameters"""
    mode: str = 'specific_shot'  # 'specific_shot' or 'shots_as_factor'
    shot_value: int = 0  # 0 or 5, only used when mode='specific_shot'
    aggregation_type: Optional[str] = None  # 'simple', 'category', 'subcategory', 'individual' or None
    min_frequency_percentile: Optional[float] = None  # Percentile threshold for data balancing
    output_dir: str = '.'
    selected_mmlu_datasets: Optional[List[str]] = None  # Added field for selected MMLU datasets

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.mode not in ['specific_shot', 'shots_as_factor']:
            raise ValueError("mode must be 'specific_shot' or 'shots_as_factor'")

        if self.mode == 'specific_shot' and self.shot_value not in [0, 5]:
            raise ValueError("shot_value must be 0 or 5 when mode is 'specific_shot'")

        if self.aggregation_type not in [None, 'simple', 'category', 'subcategory', 'individual']:
            raise ValueError("aggregation_type must be None, 'simple', 'category', 'subcategory', or 'individual'")

        if self.min_frequency_percentile is not None:
            if not 0 <= self.min_frequency_percentile <= 100:
                raise ValueError("min_frequency_percentile must be between 0 and 100")

class DataProcessor:
    """Handles data loading and preprocessing"""

    def __init__(self, df: pd.DataFrame, metadata_path: Optional[str] = None):
        self.df = df
        self.metadata_df = pd.read_csv(metadata_path) if metadata_path else None
        self.base_datasets = [
            "ai2_arc.arc_challenge",
            "ai2_arc.arc_easy",
            "hellaswag",
            "openbook_qa",
            "social_iqa"
        ]

    def get_random_mmlu_datasets(self, num_datasets: int = 5, random_seed: Optional[int] = None) -> List[str]:
        """
        Randomly select MMLU datasets from the available ones

        Args:
            num_datasets: Number of MMLU datasets to select
            random_seed: Random seed for reproducibility

        Returns:
            List of selected dataset names
        """
        if random_seed is not None:
            random.seed(random_seed)

        mmlu_datasets = [ds for ds in self.df['dataset'].unique() if ds.startswith('mmlu.')]
        selected_datasets = random.sample(mmlu_datasets, min(num_datasets, len(mmlu_datasets)))
        return selected_datasets

    def filter_datasets(self, selected_mmlu_datasets: Optional[List[str]] = None):
        """
        Filter DataFrame to include only specified MMLU datasets and base datasets

        Args:
            selected_mmlu_datasets: List of MMLU datasets to include. If None, includes all datasets.
        """
        if selected_mmlu_datasets is not None:
            # Create mask for selected MMLU datasets and base datasets
            mask = (self.df['dataset'].isin(selected_mmlu_datasets) |
                    self.df['dataset'].isin(self.base_datasets))
            self.df = self.df[mask]


    def balance_data(self, df: pd.DataFrame, min_frequency_percentile: float) -> pd.DataFrame:
        """
        Balances the dataset by removing combinations that appear too infrequently.

        Args:
            df: Input DataFrame
            min_frequency_percentile: Minimum percentile for combination frequency

        Returns:
            Filtered DataFrame containing only balanced combinations
        """
        # Define features for combinations
        features = ['template', 'separator', 'enumerator', 'choices_order']
        if 'dataset_group' in df.columns:
            features.append('dataset_group')

        # Count combinations
        combination_counts = df.groupby(features).size().reset_index(name='count')

        # Calculate threshold
        threshold = np.percentile(combination_counts['count'], min_frequency_percentile)

        # Filter combinations that appear frequently enough
        threshold = 100 if threshold < 100 else threshold
        frequent_combinations = combination_counts[combination_counts['count'] > threshold]

        # Filter original dataframe to keep only frequent combinations
        filtered_df = df.merge(
            frequent_combinations[features].drop_duplicates(),
            on=features,
            how='inner'
        )

        # Print balancing statistics
        total_combinations = len(combination_counts)
        kept_combinations = len(frequent_combinations)
        total_rows = len(df)
        kept_rows = len(filtered_df)

        print(f"\nBalancing Statistics:")
        print(f"Combinations: {kept_combinations}/{total_combinations} "
              f"({kept_combinations / total_combinations * 100:.1f}%)")
        print(f"Rows: {kept_rows}/{total_rows} ({kept_rows / total_rows * 100:.1f}%)")
        print(f"Frequency threshold: {threshold}")
        self.df = filtered_df
        # return filtered_df

    def filter_by_shots(self, shot_value: int) -> pd.DataFrame:
        """Filter data by shot value"""
        filtered_df = self.df.copy()
        if shot_value == 0:
            filtered_df = filtered_df[filtered_df["shots"] == 0]
        elif shot_value == 5:
            filtered_df = filtered_df[~filtered_df.choices_order.isin(["correct_first", "correct_last"])]
            filtered_df = filtered_df[filtered_df["shots"] == 5]
        self.df = filtered_df

    def aggregate_mmlu_data(self, aggregation_type: str,
                            selected_mmlu_datasets: Optional[List[str]] = None) -> pd.DataFrame:
        """Aggregate MMLU data based on specified type"""
        if not self.metadata_df is not None:
            raise ValueError("Metadata file required for MMLU aggregation")

        result_df = self.df.copy()

        # Extract dataset names
        result_df['dataset_name'] = result_df['dataset'].apply(
            lambda x: x.split('.')[-1] if x.startswith('mmlu.') else x
        )

        # Create mappings
        category_mapping = dict(zip(self.metadata_df['Name'], self.metadata_df['Category']))
        subcategory_mapping = dict(zip(self.metadata_df['Name'], self.metadata_df['Sub_Category']))

        if aggregation_type == 'simple':
            if selected_mmlu_datasets is not None:
                # Case: Selected MMLU datasets
                result_df['dataset_group'] = result_df['dataset'].apply(
                    lambda x: 'Selected MMLU' if x in selected_mmlu_datasets else 'Other'
                )
            else:
                # Case: Regular MMLU aggregation
                result_df['dataset_group'] = result_df['dataset'].apply(
                    lambda x: 'MMLU' if x.startswith('mmlu.') else x
                )
                # result_df['dataset_group'] = result_df['dataset'].apply(
                #     lambda x: 'MMLU' if x.startswith('mmlu.') else
                #     'MMLU Pro' if x.startswith('mmlu_pro.') else x
                # )
        elif aggregation_type == 'individual':
            if selected_mmlu_datasets is not None:
                # Case: Handle individual selected datasets
                result_df['dataset_group'] = result_df['dataset'].apply(
                    lambda x: x if x in selected_mmlu_datasets else 'Other'
                )
            else:
                # Case: No selected datasets, treat each MMLU dataset individually
                result_df['dataset_group'] = result_df['dataset'].apply(
                    lambda x: x if x.startswith('mmlu.') else 'Other'
                )
        elif aggregation_type == 'category':
            result_df['dataset_group'] = result_df.apply(
                lambda row: category_mapping.get(row['dataset_name'], 'Other')
                if row['dataset'].startswith('mmlu.') else 'Other', axis=1
            )
        elif aggregation_type == 'subcategory':
            result_df['dataset_group'] = result_df.apply(
                lambda row: subcategory_mapping.get(row['dataset_name'], 'Other')
                if row['dataset'].startswith('mmlu.') else 'Other', axis=1
            )

        # Aggregate accuracy by the new grouping
        aggregated_df = result_df.groupby([
            'model_name', 'dataset_group', 'template', 'separator',
            'enumerator', 'choices_order', 'shots'
        ])['accuracy'].mean().reset_index()

        self.df = aggregated_df


    def aggregate_mmlu_data1(self, aggregation_type: str,
                            selected_mmlu_datasets: Optional[List[str]] = None) -> pd.DataFrame:
        """Aggregate MMLU data based on specified type"""
        if not self.metadata_df is not None:
            raise ValueError("Metadata file required for MMLU aggregation")

        result_df = self.df.copy()

        # Extract dataset names
        result_df['dataset_name'] = result_df['dataset'].apply(
            lambda x: x.split('.')[-1] if x.startswith('mmlu.') else x
        )

        # Create mappings
        category_mapping = dict(zip(self.metadata_df['Name'], self.metadata_df['Category']))
        subcategory_mapping = dict(zip(self.metadata_df['Name'], self.metadata_df['Sub_Category']))

        if aggregation_type == 'simple':
            result_df['dataset_group'] = result_df['dataset'].apply(
                lambda x: 'Selected MMLU' if (x.startswith('mmlu.') and
                                              (selected_mmlu_datasets is None or x in selected_mmlu_datasets))
                else 'Other'
            )
        elif aggregation_type == 'individual':
            # Treat each selected MMLU dataset as its own group
            result_df['dataset_group'] = result_df['dataset'].apply(
                lambda x: x if (x.startswith('mmlu.') and x in selected_mmlu_datasets) else 'Other'
            )
        elif aggregation_type == 'category':
            result_df['dataset_group'] = result_df.apply(
                lambda row: category_mapping.get(row['dataset_name'], 'Other')
                if row['dataset'].startswith('mmlu.') else 'Other', axis=1
            )
        elif aggregation_type == 'subcategory':
            result_df['dataset_group'] = result_df.apply(
                lambda row: subcategory_mapping.get(row['dataset_name'], 'Other')
                if row['dataset'].startswith('mmlu.') else 'Other', axis=1
            )

        # Aggregate accuracy by the new grouping
        aggregated_df = result_df.groupby([
            'model_name', 'dataset_group', 'template', 'separator',
            'enumerator', 'choices_order', 'shots'
        ])['accuracy'].mean().reset_index()

        self.df = aggregated_df

class StatisticalAnalyzer:
    """Handles statistical analysis"""

    @staticmethod
    def analyze_factor_significance(df: pd.DataFrame, factor: str) -> Dict:
        """Analyze significance of a single factor"""
        formula = f"accuracy ~ C({factor})"
        model = smf.ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        f_stat = anova_table.iloc[0]['F']
        p_value = anova_table.iloc[0]['PR(>F)']
        ss_effect = anova_table.iloc[0]['sum_sq']
        ss_error = anova_table.iloc[1]['sum_sq']
        partial_eta_sq = ss_effect / (ss_effect + ss_error)

        return {
            'F_statistic': f_stat,
            'p_value': p_value,
            'partial_eta_squared': partial_eta_sq,
            'sample_size': len(df),
            'effect_magnitude': StatisticalAnalyzer._classify_effect_size(partial_eta_sq),
            'practically_significant': partial_eta_sq >= 0.01
        }

    @staticmethod
    def _classify_effect_size(eta_squared: float) -> str:
        """Classify effect size based on partial eta squared"""
        if eta_squared < 0.01:
            return "Negligible"
        elif eta_squared < 0.06:
            return "Small"
        elif eta_squared < 0.14:
            return "Medium"
        return "Large"


class Visualizer:
    """Handles visualization of results"""

    @staticmethod
    def generate_output_filename(config: 'AnalysisConfig') -> str:
        """
        Generate informative filename based on analysis configuration
        """
        parts = []

        # Add mode and shot info
        if config.mode == 'specific_shot':
            parts.append(f"shot{config.shot_value}")
        else:
            parts.append("shots_as_factor")

        # Add aggregation type if used
        if config.aggregation_type:
            parts.append(config.aggregation_type)

        # Add sampling info if used
        if config.selected_mmlu_datasets is not None:
            parts.append("sampled")

        # Add balancing info if used
        if config.min_frequency_percentile is not None:
            parts.append(f"balanced{int(config.min_frequency_percentile)}")

        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d')
        parts.append(timestamp)

        return f"effect_sizes_{'_'.join(parts)}.png"

    @staticmethod
    def plot_effect_sizes(results_df: pd.DataFrame, config: 'AnalysisConfig'):
        """Create bar plot of effect sizes with informative title and filename"""
        plt.figure(figsize=(15, 8))
        sns.barplot(data=results_df, x='Factor', y='partial_eta_squared', hue='Model')

        # Create informative title
        title_parts = []
        if config.mode == 'specific_shot':
            title_parts.append(f"{config.shot_value}-Shot Analysis")
        else:
            title_parts.append("Shots as Factor Analysis")

        if config.aggregation_type:
            aggregation_desc = config.aggregation_type.title()
            if config.selected_mmlu_datasets is not None:
                aggregation_desc = f"Sampled MMLU {aggregation_desc}"
            title_parts.append(f"{aggregation_desc} Aggregation")

        if config.min_frequency_percentile is not None:
            title_parts.append(f"Balanced (p{int(config.min_frequency_percentile)})")

        plt.title(' - '.join(title_parts))
        plt.xlabel('Factor')
        plt.ylabel('Partial Eta Squared')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Generate filename and save
        filename = Visualizer.generate_output_filename(config)
        output_path = os.path.join(config.output_dir, filename)
        os.makedirs(config.output_dir, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        print(f"\nSaved plot to: {output_path}")


class AnalysisPipeline:
    """Main analysis pipeline"""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def run_analysis(self, df: pd.DataFrame, metadata_path: Optional[str] = None) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series]:
        """Run complete analysis pipeline"""
        # Initialize processor
        processor = DataProcessor(df, metadata_path)
        # Filter datasets if using sampled MMLU datasets
        if self.config.selected_mmlu_datasets is not None:
            processor.filter_datasets(self.config.selected_mmlu_datasets)

        # Process data based on configuration
        # Apply shot filtering only if we're in specific_shot mode
        if self.config.mode == 'specific_shot':
            processor.filter_by_shots(self.config.shot_value)

        # Apply MMLU aggregation if specified
        if self.config.aggregation_type:
            processor.aggregate_mmlu_data(self.config.aggregation_type)

        # Balance data if specified
        if self.config.min_frequency_percentile is not None:
            processor.balance_data(processor.df, self.config.min_frequency_percentile)
        processed_df = processor.df
        # Run analysis for each model
        analyzer = StatisticalAnalyzer()
        factors = ["template", "separator", "enumerator", "choices_order"]
        if self.config.mode == 'shots_as_factor':
            factors.append("shots")

        all_results = []
        for model_name in processed_df['model_name'].unique():
            model_df = processed_df[processed_df['model_name'] == model_name]
            if len(model_df) >= 1000:
                model_results = []
                for factor in factors:
                    result = analyzer.analyze_factor_significance(model_df, factor)
                    result.update({'Model': model_name, 'Factor': factor})
                    model_results.append(result)
                all_results.extend(model_results)

        # Combine and summarize results
        combined_results = pd.DataFrame(all_results)
        summary = self._create_summary(combined_results)
        practical_effects = self._count_practical_effects(combined_results)

        # Create visualization
        visualizer = Visualizer()
        visualizer.plot_effect_sizes(combined_results, self.config)

        return combined_results, summary, practical_effects

    def _create_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics"""
        return results_df.groupby('Factor').agg({
            'partial_eta_squared': ['mean', 'std', 'min', 'max'],
            'practically_significant': 'sum'
        }).round(4)

    def _count_practical_effects(self, results_df: pd.DataFrame) -> pd.Series:
        """Count practically significant effects by model"""
        return results_df[results_df['practically_significant']].groupby('Model').size()



def process_config(config, df, metadata_path):
    pipeline = AnalysisPipeline(config)
    # הפונקציה תחזיר את מה שמעניין אותנו מכל הרצה
    results, summary, effects = pipeline.run_analysis(df, metadata_path)
    return config, results, summary, effects

if __name__ == "__main__":
    # Load data
    df = pd.read_parquet("aggregated_results.parquet")
    # df = df[~df.model_name.startswith("mmlu_pro")]  # Filter out random models
    # df = df[~df["dataset"].str.startswith("mmlu_pro")]

    metadata_path = "/Users/ehabba/PycharmProjects/LLM-Evaluation/Data/mmlu_metadata.csv"

    # Configure analysis
    from dataclasses import dataclass
    from typing import List
    from itertools import product
    from tqdm import tqdm

    # נגדיר תסריטים (scenario_name -> aggregation_type)
    SCENARIOS = {
        "baseline": None,  # Non-MMLU
        "mmlu_simple": "simple",  # MMLU vs MMLU Pro vs Other
        "mmlu_category": "category",
        "mmlu_subcategory": "subcategory",
        "mmlu_sampled_simple": "simple",  # New: Sampled MMLU datasets combined
        "mmlu_sampled_individual": "individual",  # New: Each sampled MMLU dataset separately
    }

    # נגדיר אפשרויות למצב השוט:
    # נכניס רק את האפשרויות שבאמת רלוונטיות לנו
    SHOT_SETTINGS = [
        ("specific_shot", 0, "0shot"),
        ("specific_shot", 5, "5shot"),
        ("shots_as_factor", None, "shots_as_factor"),
    ]

    # נגדיר אפשרות לרוץ גם רגיל וגם מאוזן
    BALANCED_OPTIONS = [False, True]

    all_configs = []

    # Get random MMLU datasets (do this once to use the same datasets for all configs)
    processor = DataProcessor(df)
    selected_mmlu_datasets = processor.get_random_mmlu_datasets(num_datasets=5, random_seed=42)

    for scenario_name, agg_type in SCENARIOS.items():
        for mode, shot_value, shot_label in SHOT_SETTINGS:
            for is_balanced in BALANCED_OPTIONS:
                # Set output directory based on is_balanced
                if is_balanced:
                    output_dir = f"results/balanced/{scenario_name}"
                    min_freq_percentile = 50
                else:
                    output_dir = f"results/{scenario_name}"
                    min_freq_percentile = None

                # Handle shots_as_factor mode
                safe_shot_value = 0 if mode == 'shots_as_factor' else shot_value

                # Add selected_mmlu_datasets to config for sampled scenarios
                extra_params = {}
                if scenario_name in ['mmlu_sampled_simple', 'mmlu_sampled_individual']:
                    extra_params['selected_mmlu_datasets'] = selected_mmlu_datasets

                config = AnalysisConfig(
                    mode=mode,
                    shot_value=safe_shot_value,
                    aggregation_type=agg_type,
                    min_frequency_percentile=min_freq_percentile,
                    output_dir=output_dir,
                    **extra_params
                )
                all_configs.append(config)

    futures = []
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=4) as executor:
    # with ProcessPoolExecutor() as executor:
        # שולחים כל קונפיגורציה לריצה מקבילית
        for config in all_configs:
            future = executor.submit(process_config, config, df, metadata_path)
            futures.append(future)

        # איסוף התוצאות תוך כדי מדידת התקדמות ב-tqdm
        for future in tqdm(as_completed(futures), total=len(futures)):
            config, results, summary, effects = future.result()
