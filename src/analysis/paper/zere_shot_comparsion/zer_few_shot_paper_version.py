import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

USE_SAMPLED_DATA = False


class DataProcessor:
    def __init__(self, base_dir: str, models: List[str], datasets: List[str]):
        self.base_dir = base_dir
        self.models = models
        self.datasets = datasets
        self.required_columns = [
            'dataset', 'sample_index', 'model',
            'shots', 'template', 'separator', 'enumerator', 'choices_order',
            'score'
        ]

    def load_and_process_data(self) -> pd.DataFrame:
        """
        Load and process data for all models with specific column requirements and filtering.
        """
        all_data = []

        for model in tqdm(self.models):
            print(f"Processing data for model: {model}")
            model_name = model.split('/')[-1]
            file_path = os.path.join(self.base_dir, f"results_{model_name}.parquet")

            if os.path.exists(file_path):
                # Load data with specific columns
                if USE_SAMPLED_DATA and os.path.exists(f"results_{model_name}_sampled.parquet"):
                    data = pd.read_parquet(f"results_{model_name}_sampled.parquet")
                else:
                    data = pd.read_parquet(file_path, columns=self.required_columns)
                # Filter for wanted datasets
                data = data[data['dataset'].isin(self.datasets)]

                # Remove specific choices_order
                # for all shots=5 remove correct_first and correct_last
                data = data[~((data['shots'] == 5) & (data['choices_order'].isin(["correct_first", "correct_last"])))]
                # take 10k
                if USE_SAMPLED_DATA:
                    data = data.sample(n=1000, random_state=1, replace=True)
                    data.to_parquet(f"results_{model_name}_sampled.parquet")
                # remove duplicate rows with same shots, template, separator, enumerator, choices_order, model, dataset, sample_index
                data = data.drop_duplicates(
                    subset=['shots', 'template', 'separator', 'enumerator', 'choices_order', 'model', 'dataset',
                            'sample_index'])
                rankings = self._compute_rankings(data)
                print(
                    f"Model {model_name}: Computed rankings for {len(rankings)} configurations with {len(data)} samples")
                print(rankings['dataset'].value_counts())
                # Add model information

                all_data.append(rankings)
            else:
                print(f"Warning: Could not find data for {model} at {file_path}")

        if not all_data:
            raise ValueError("No data was loaded for any model")

        combined_data = pd.concat(all_data, ignore_index=True)

        # Compute rankings and apply filters

        # Filter based on count
        # rankings = rankings[rankings['count'] > 1000]

        # Merge rankings back with original data
        # final_data = self._merge_rankings(combined_data, rankings)

        return combined_data

    def _compute_rankings(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rankings for configurations.
        """
        # Group by relevant columns and compute statistics
        rankings = data.groupby(
            ['shots', 'template', 'separator', 'enumerator', 'choices_order', 'model', 'dataset']
        ).agg({
            'score': ['mean', 'std', 'count']
        }).reset_index()

        # Flatten column names
        rankings.columns = ['shots', 'template', 'separator', 'enumerator', 'choices_order',
                            'model', 'dataset', 'accuracy', 'std', 'count']

        # remove row with count < 90
        # rankings = rankings[rankings['count'] > 90]
        return rankings

    def _merge_rankings(self, data: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
        """
        Merge rankings back with original data.
        """
        merge_cols = ['shots', 'template', 'separator', 'enumerator', 'choices_order',
                      'model', 'dataset']
        return pd.merge(data, rankings, on=merge_cols, how='inner')


class Visualizer:
    def __init__(self):
        self.color_scheme = {
            'meta-llama/Llama-3.2-1B-Instruct': "#1f77b4",  # Blue
            'allenai/OLMoE-1B-7B-0924-Instruct': "#ff7f0e",  # Orange
            'meta-llama/Meta-Llama-3-8B-Instruct': "#2ca02c",  # Green
            'meta-llama/Llama-3.2-3B-Instruct': "#d62728",  # Red
            'mistralai/Mistral-7B-Instruct-v0.3': "#9467bd"  # Purple
        }

    def _wrap_labels(self, text, width=12):
        if text == 'ai2_arc.arc_challenge':
            text = "ARC Challenge"
        if text == 'hellaswag':
            text = "HellaSwag"
        if text == 'openbook_qa':
            text = "OpenBookQA"
        if text == 'social_iqa':
            text = "Social IQA"
        return text

    def create_grid_shot_comparison(self, data, models):
        # Set up the plotting style
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        axes = axes.flatten()  # Flatten to make iteration easier

        # Width for jittering points
        jitter_width = 0.2

        # Order of datasets for consistency
        order_of_the_labels = [
            'mmlu.college_biology', 'mmlu_pro.law',
            'ai2_arc.arc_challenge', 'social_iqa',
            'hellaswag', 'openbook_qa'
        ]
        order_of_the_labels = ['mmlu.college_biology', 'hellaswag', 'ai2_arc.arc_challenge', 'social_iqa',
                               'openbook_qa', 'mmlu_pro.law', ]
        # Process each model in a separate subplot
        for idx, (ax, model) in enumerate(zip(axes, models)):
            # Filter data for current model
            model_data = data[data['model'] == model]
            zero_shot = model_data[model_data['shots'] == 0]
            five_shot = model_data[model_data['shots'] == 5]

            # Get datasets present in the data
            datasets = sorted(model_data['dataset'].unique())
            sorted_labels = [x for x in order_of_the_labels if x in datasets]

            # Plot points for each dataset
            for i, dataset in enumerate(sorted_labels):
                # Get accuracy values
                zero_values = zero_shot[zero_shot['dataset'] == dataset]['accuracy'] * 100
                five_values = five_shot[five_shot['dataset'] == dataset]['accuracy'] * 100

                # Create jittered positions
                zero_x = np.random.normal(i - jitter_width / 2, jitter_width / 4, size=len(zero_values))
                five_x = np.random.normal(i + jitter_width / 2, jitter_width / 4, size=len(five_values))

                # Plot scatter points
                ax.scatter(zero_x, zero_values, color='red', alpha=0.6, s=50,
                           label='Zero-shot' if i == 0 else "")
                ax.scatter(five_x, five_values, color='blue', alpha=0.6, s=50,
                           label='Five-shot' if i == 0 else "")

                # Add boxplots
                bp_zero = ax.boxplot([zero_values], positions=[i - jitter_width / 2],
                                     widths=jitter_width / 2, patch_artist=True,
                                     medianprops=dict(color='black'),
                                     flierprops=dict(marker='none'),
                                     boxprops=dict(facecolor='red', alpha=0.3),
                                     showfliers=False)

                bp_five = ax.boxplot([five_values], positions=[i + jitter_width / 2],
                                     widths=jitter_width / 2, patch_artist=True,
                                     medianprops=dict(color='black'),
                                     flierprops=dict(marker='none'),
                                     boxprops=dict(facecolor='blue', alpha=0.3),
                                     showfliers=False)
                ax.grid(True, axis='y', linestyle='--', alpha=0.5)

            # Customize subplot
            ax.set_ylim(0, 100)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_xlim(-0.5, len(sorted_labels) - 0.5)

            # Format model name for title
            model_name_title = model.split('/')[-1]
            model_name_title = model_name_title.replace('Meta-Llama', 'Llama')
            ax.set_title(model_name_title, fontsize=21, fontfamily='DejaVu Serif', pad=15)

            # Create dataset labels
            dataset_labels = []
            for dataset in sorted_labels:
                if dataset == 'ai2_arc.arc_challenge':
                    dataset_labels.append('ARC\nChallenge')
                elif dataset == 'hellaswag':
                    dataset_labels.append('HellaSwag')
                elif dataset == 'openbook_qa':
                    dataset_labels.append('OpenBook-\nQA')
                elif dataset == 'social_iqa':
                    dataset_labels.append('Social\nIQa')
                elif dataset == 'mmlu.college_biology':
                    dataset_labels.append('MMLU\ncollege biology')
                elif dataset == 'mmlu_pro.law':
                    dataset_labels.append('MMLU-Pro\nlaw')
                else:
                    dataset_labels.append(dataset)

            # Set axis labels and ticks
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(dataset_labels, rotation=0, ha='center',
                               fontsize=17, fontfamily='DejaVu Serif')
            # Configure y-axis ticks
            ax.tick_params(axis='y', labelsize=16)

            # Hide y-axis labels for right subplots (idx 1 and 3)
            if idx > 0:
                ax.tick_params(labelleft=False)  # This hides only the labels but leaves the ticks (and grid) intact.
                # ax.yaxis.set_ticklabels([])
                # ax.yaxis.set_ticks([])  # מסיר גם את הקווים וגם את התוויות
            if idx > 0:
                ax.tick_params(axis='y', length=0, labelleft=False)
        # Add legend only to the first subplot
            if idx == 1:
                ax.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, 1.35),
                          ncol=2, frameon=True, edgecolor='black', markerscale=3.0)
            # Adjust subplot spacing
        plt.subplots_adjust(left=0.1)  # Increase left margin

        # Add common y-label with adjusted position
        fig.text(-0.0, 0.5, 'Accuracy Score', va='center', rotation='vertical',
                 fontsize=18, fontfamily='DejaVu Serif')

        # Adjust layout with specific padding
        plt.tight_layout(pad=1.5)
        # Adjust layout
        plt.tight_layout()

        # Save plot
        plt.savefig('shot_comparison_grid.png', dpi=600, bbox_inches='tight')
        plt.savefig('shot_comparison_grid.svg', bbox_inches='tight')
        plt.close()


def main():
    # Configuration
    base_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/ConfigurationOptimizer/data"

    # Define models
    models = [
        # 'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        # 'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3'
    ]

    # Define datasets
    datasets = [
        "ai2_arc.arc_challenge",
        # "ai2_arc.arc_easy",
        "hellaswag",
        # "openbook_qa",
        "social_iqa",
        "mmlu.college_biology",
        "mmlu_pro.law"
    ]

    # Process data
    processor = DataProcessor(base_dir, models, datasets)
    data = processor.load_and_process_data()

    # Create visualization
    visualizer = Visualizer()
    visualizer.create_grid_shot_comparison(data, models)


if __name__ == "__main__":
    main()
