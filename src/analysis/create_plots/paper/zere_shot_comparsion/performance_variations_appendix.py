import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
USE_SAMPLED_DATA = True

interesting_datasets2 = [
    "ai2_arc.arc_challenge",
    "ai2_arc.arc_easy",
    "hellaswag",
    "openbook_qa",
    "social_iqa",
]

subtasks = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
pro_subtuask = [
    "history",
    "law",
    "health",
    "physics",
    "business",
    "other",
    "philosophy",
    "psychology",
    "economics",
    "math",
    "biology",
    "chemistry",
    "computer_science",
    "engineering",
]
interesting_datasets = []
interesting_datasets.extend(["mmlu." + name for name in subtasks])
interesting_datasets.extend(["mmlu_pro." + name for name in pro_subtuask])
interesting_datasets.extend(interesting_datasets2)


def format_dataset_string(dataset_string):
    mapping = {
        'ai2_arc.arc_challenge': "ARC\nChallenge",
        'ai2_arc.arc_easy': "ARC\nEasy",
        'hellaswag': "HellaSwag",
        'openbook_qa': "OpenBookQA",
        'social_iqa': "Social\nIQa",
        # 'mmlu.college_biology': "MMLU\nCollege Biology",
        # 'mmlu_pro.law': "MMLU-Pro\nLaw"
    }
    if dataset_string in mapping:
        return mapping[dataset_string]
    if "." not in dataset_string:
        return dataset_string
    dataset, category = dataset_string.split('.')  # מחלק את השם ל"דאטהסט" ו"קטגוריה"

    # מחלק את שם הדאטהסט לפי "_" ומוסיף ".\n" אחרי כל חלק
    dataset_parts = dataset.split('_')
    formatted_dataset = '.\n'.join(dataset_parts) + '.\n'

    # מחלק את הקטגוריה לפי "_" ומוסיף "-\n" אחרי כל חלק
    category_parts = category.split('_')
    if     "macroeconomics" in category_parts:
        del category_parts[category_parts.index("macroeconomics")]
        category_parts.append("macro")
        category_parts.append("economics")

    if "microeconomics" in category_parts:
        del category_parts[category_parts.index("microeconomics")]
        category_parts.append("micro")
        category_parts.append("economics")
    if  "mathematics" in category_parts:
        del category_parts[category_parts.index("mathematics")]
        category_parts.append("math")
        category_parts.append("ematics")
    if "international" in category_parts:
        del category_parts[category_parts.index("international")]
        category_parts.append("inter")
        category_parts.append("national")
    if "jurisprudence" in category_parts:
        del category_parts[category_parts.index("jurisprudence")]
        category_parts.append("juris")
        category_parts.append("prudence")
    if "professional" in category_parts:
        del category_parts[category_parts.index("professional")]
        category_parts.append("pro")
        category_parts.append("fessional")
    if "econometrics" in category_parts:
        del category_parts[category_parts.index("econometrics")]
        category_parts.append("econ")
        category_parts.append("ometrics")


    formatted_category = '-\n'.join(category_parts)

    return f"{formatted_dataset}{formatted_category}"

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
                    # sample with reapeat 1000
                    data = data.sample(n=10000, replace=True, random_state=42)
                    # save th dummt data
                    data.to_parquet(f"results_{model_name}_sampled.parquet")

                # remove duplicate rows with same shots, template, separator, enumerator, choices_order, model, dataset, sample_index
                data = data.drop_duplicates(subset=['shots', 'template', 'separator', 'enumerator', 'choices_order', 'model', 'dataset', 'sample_index'])
                rankings = self._compute_rankings(data)
                print(f"Model {model_name}: Computed rankings for {len(rankings)} configurations with {len(data)} samples")
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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EnhancedVisualizer:
    def __init__(self):
        self.color_scheme = {
            'meta-llama/Llama-3.2-1B-Instruct': "#1f77b4",  # Blue
            'allenai/OLMoE-1B-7B-0924-Instruct': "#ff7f0e",  # Orange
            'meta-llama/Meta-Llama-3-8B-Instruct': "#2ca02c",  # Green
            'meta-llama/Llama-3.2-3B-Instruct': "#d62728",  # Red
            'mistralai/Mistral-7B-Instruct-v0.3': "#9467bd"  # Purple
        }

    def get_display_name(self, dataset_name):
        """
        Convert dataset name to display format
        """
        if dataset_name in interesting_datasets2:
            text = dataset_name
            if text == 'ai2_arc.arc_challenge':
                text = "ARC\nChallenge"
            if text == 'hellaswag':
                text = "HellaSwag"
            if text == 'openbook_qa':
                text = "OpenBookQA"
            if text == 'social_iqa':
                text = "Social\nIQa"
            return text
        if dataset_name.startswith('mmlu.'):
            return f"MMLU\n{dataset_name.split('.')[-1].replace('_', ' ')}"
        elif dataset_name.startswith('mmlu_pro.'):
            return f"MMLU-Pro\n{dataset_name.split('.')[-1]}"
        return dataset_name
    def _wrap_labels(self, text):
        mapping = {
            'ai2_arc.arc_challenge': "ARC\nChallenge",
            'ai2_arc.arc_easy': "ARC\nEasy",
            'hellaswag': "HellaSwag",
            'openbook_qa': "OpenBookQA",
            'social_iqa': "Social\nIQa",
            # 'mmlu.college_biology': "MMLU\nCollege Biology",
            # 'mmlu_pro.law': "MMLU-Pro\nLaw"
        }
        return mapping.get(text, text)

    def _create_lighter_color(self, hex_color):
        # Convert hex to RGB
        rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        # Make lighter by mixing with white, but keep more of original color
        lighter_rgb = tuple(int(c + (255 - c) * 0.35) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)

    def _create_darker_color(self, hex_color):
        # Convert hex to RGB
        rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        # Make darker by reducing RGB values
        darker_rgb = tuple(int(c * 0.7) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*darker_rgb)

    def create_comparison_plot(self, data: pd.DataFrame, i,dataset_chunks,models):
        # Filter for shots = 0 and 5
        # Sort datasets and models
        model_order = models
        filtered_data = data

        fig, ax = plt.subplots(figsize=(36, 5.2))

        # Setup dimensions
        dataset_order = dataset_chunks
        num_datasets = len(dataset_chunks)
        num_models = len(model_order)

        # Constants for positioning
        model_group_width = 0.18  # Width for each model's group (including both shots)
        shot_width = 0.06  # Width for each shot's data
        model_spacing = 0.05  # Space between different models

        # Add extra space between datasets by modifying the dataset center
        dataset_spacing = 1.6  # Multiplier for space between datasets

        legend_handles = []

        # Plot points for each dataset
        for dataset_idx, dataset in enumerate(dataset_order):
            dataset_center = dataset_idx * dataset_spacing

            # Plot each model
            for model_idx, model in enumerate(model_order):
                # Calculate center position for this model's group
                model_center = dataset_center + (model_idx - num_models / 2) * (model_group_width + model_spacing)
                model_center = dataset_center + (model_idx - (num_models - 1) / 2) * (model_group_width + model_spacing)

                model_data = filtered_data[
                    (filtered_data['model'] == model) &
                    (filtered_data['dataset'] == dataset)
                    ]

                if len(model_data) == 0:
                    continue

                # Calculate position for this shot
                shot_center = model_center

                # Get colors
                base_color = self.color_scheme[model]
                color = self._create_lighter_color(base_color)

                # Add jitter to x positions
                x = np.random.normal(shot_center, shot_width * 0.2, size=len(model_data))
                y = model_data['accuracy'].values * 100

                # Plot points
                ax.scatter(x, y,
                           alpha=0.6,
                           s=16,
                           c=[color],
                           zorder=2)

                # Add boxplot
                bp = ax.boxplot([y],
                                positions=[shot_center],
                                widths=shot_width * 0.8,
                                patch_artist=True,
                                medianprops=dict(color='black'),
                                flierprops=dict(marker='none'),
                                boxprops=dict(facecolor='white',
                                              color=color,
                                              alpha=0.3),
                                whiskerprops=dict(color=color),
                                capprops=dict(color=color),
                                showfliers=False)

                # Add to legend (only once per model-shot combination)
                if dataset_idx == 0 and i==0:
                    legend_handles.append(plt.Line2D([0], [0],
                                                     marker='o',
                                                     color='w',
                                                     markerfacecolor=color,
                                                     markersize=10,
                                                     label=f'{model.split("/")[-1]}'))

            # Add subtle vertical line between datasets
            ax.axvline(x=dataset_idx + 0.5, color='gray', linestyle=':', alpha=0.3)

        # Customize plot
        ax.set_ylabel('Accuracy Score (%)', fontsize=28, fontfamily='DejaVu Serif', labelpad=15)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, (num_datasets - 1) * dataset_spacing + 0.5)

        # Set x-ticks and labels
        plt.xticks(np.arange(num_datasets) * dataset_spacing,
                   [format_dataset_string(label) for label in dataset_order],
                   rotation=0,
                   ha='center',
                   fontsize=32,
                   fontfamily='DejaVu Serif')
        plt.yticks(fontsize=29, fontfamily='DejaVu Serif')

        # Add legend
        if i==0:
            legend = ax.legend(handles=legend_handles,
                               bbox_to_anchor=(0.5, 1.15),
                               loc='center',
                               ncol=5,
                               borderaxespad=0.,
                               frameon=True,
                               edgecolor='black',
                               fancybox=False,
                               fontsize=26,
                               markerscale=2.6)  #

        # Adjust layout
        plt.subplots_adjust(top=0.85)
        ax.tick_params(axis='x', pad=20)

        # Save plot
        plt.savefig(f'model_performance_comparison_agg_shots_{i}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'model_performance_comparison_agg_shots_{i}.pdf', bbox_inches='tight')
        plt.close()


def main():
    # Define configurations
    base_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/ConfigurationOptimizer/data"

    models = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3'
    ]



    # Process data

    # Create visualization
    visualizer = EnhancedVisualizer()

    # Split datasets into chunks
    processor = DataProcessor(base_dir, models, interesting_datasets)
    data = processor.load_and_process_data()
    avilable_datasets = data['dataset'].unique()
    avilable_datasets = data['dataset'].unique()
    not_mmlu = [dataset for dataset in avilable_datasets if not dataset.startswith('mmlu')]
    mmlu = [dataset for dataset in avilable_datasets if
            dataset.startswith('mmlu') and not dataset.startswith('mmlu_pro')]
    mmlu_pro = [dataset for dataset in avilable_datasets if dataset.startswith('mmlu_pro')]
    # take random 1 mmlu pro datasets
    chosen_mmlu_pro = np.random.choice(mmlu_pro, 1)
    # take random 3 mmlu datasets
    chosen_mmlu = np.random.choice(mmlu, 3)
    chosen_datasets = [*not_mmlu, *chosen_mmlu, *chosen_mmlu_pro ,"mmlu_pro.law"]
    avilable_datasets = chosen_datasets

    dataset_chunks = [avilable_datasets[i:i + 13]
                      for i in range(0, len(avilable_datasets), 13)]

    for i, chunk in enumerate(dataset_chunks):
        data_chunk = data[data['dataset'].isin(chunk)]
        visualizer.create_comparison_plot(data  = data_chunk, i=i,dataset_chunks=chunk, models=models)
        break

if __name__ == "__main__":
    main()

