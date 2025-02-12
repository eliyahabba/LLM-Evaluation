import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


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
                data = pd.read_parquet(file_path, columns=self.required_columns)

                # Filter for wanted datasets
                data = data[data['dataset'].isin(self.datasets)]

                # Remove specific choices_order
                # for all shots=5 remove correct_first and correct_last
                data = data[~((data['shots'] == 5) & (data['choices_order'].isin(["correct_first", "correct_last"])))]
                # take 10k
                data = data.sample(n=10000, random_state=1)
                # remove duplicate rows with same shots, template, separator, enumerator, choices_order, model, dataset, sample_index
                # data = data.drop_duplicates(subset=['shots', 'template', 'separator', 'enumerator', 'choices_order', 'model', 'dataset', 'sample_index'])
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


class Visualizer:
    def __init__(self):
        self.color_scheme = {
            'meta-llama/Llama-3.2-1B-Instruct': "#1f77b4",  # Blue
            'allenai/OLMoE-1B-7B-0924-Instruct': "#ff7f0e",  # Orange
            'meta-llama/Meta-Llama-3-8B-Instruct': "#2ca02c",  # Green
            'meta-llama/Llama-3.2-3B-Instruct': "#d62728",  # Red
            'mistralai/Mistral-7B-Instruct-v0.3': "#9467bd"  # Purple
        }

        # Set the style for matplotlib

    def _wrap_labels(self, text, width=12):
        import textwrap
        if text =='ai2_arc.arc_challenge':
            text = "ARC Challenge"
        if text == 'hellaswag':
            text = "HellaSwag"
        if text == 'openbook_qa':
            text = "OpenBookQA"
        if text == 'social_iqa':
            text = "Social IQA"
        return text
        """Wrap text labels to specified width"""
        return '\n'.join(textwrap.wrap(text, width=width))

    def save_legend_only(self, data: pd.DataFrame):
        """
        Create and save only the legend part of the plot as a separate image.
        """
        # Get model medians to determine which models to include in legend
        model_medians = data.groupby('model')['accuracy'].median().sort_values(ascending=False)

        # Create a new figure for legend only
        figleg = plt.figure(figsize=(16, 1))  # Adjust size as needed
        ax = figleg.add_subplot(111)

        # Create legend elements
        legend_elements = [plt.Line2D([0], [0],
                                      marker='s',
                                      color='w',
                                      markerfacecolor=self.color_scheme.get(model_name),
                                      markersize=14,
                                      label=model_name.split('/')[-1])
                           for model_name in model_medians.index]

        # Create the legend
        legend = ax.legend(handles=legend_elements,
                           loc='center',
                           ncol=2,
        borderaxespad = 0., frameon = True,
        edgecolor = 'black',
        fancybox = False)

        # Set figure background to white
        figleg.patch.set_facecolor('white')

        # Save legend only with white background
        plt.savefig('legend_only.png',
                    dpi=300,
                    bbox_inches='tight',
                    transparent=False,
                    facecolor='white',
                    edgecolor='none')
        plt.savefig('legend_only.pdf',
                    bbox_inches='tight',
                    transparent=False,
                    facecolor='white',
                    edgecolor='none')

        print("Legend saved separately in PDF and PNG formats")

        # Close the figure to free memory
        plt.close()

    def create_plot(self, data: pd.DataFrame):
        """
        Create publication-ready plot with enhanced styling using matplotlib.
        """
        # Sort models and datasets by median performance
        model_medians = data.groupby('model')['accuracy'].median().sort_values(ascending=False)
        dataset_medians = data.groupby('dataset')['accuracy'].median().sort_index()

        # Create figure and axis with specified size
        fig, ax = plt.subplots(figsize=(16, 12))

        # Calculate positions for the boxes - Reduced width for tighter spacing
        num_datasets = len(dataset_medians)
        num_models = len(model_medians)
        width = 0.6 / num_models  # Reduced from 0.8 to 0.6 for tighter spacing

        # Plot boxes for each model
        for i, (model_name, _) in enumerate(model_medians.items()):
            model_data = data[data['model'] == model_name]
            positions = np.arange(num_datasets) + (i - num_models / 2 + 0.5) * width
            bp = ax.boxplot([model_data[model_data['dataset'] == dataset]['accuracy'] * 100
                             for dataset in dataset_medians.index],
                            positions=positions,
                            widths=width * 0.8,
                            patch_artist=True,
                            medianprops=dict(color='black'),
                            flierprops=dict(marker='o',
                                            markerfacecolor=self.color_scheme.get(model_name),
                                            markersize=4,
                                            alpha=0.6),
                            boxprops=dict(facecolor='white',
                                          color=self.color_scheme.get(model_name),
                                          alpha=0.3),
                            whiskerprops=dict(color=self.color_scheme.get(model_name)),
                            capprops=dict(color=self.color_scheme.get(model_name)))

            # Add individual points (jittered) with reduced jitter
            for j, dataset in enumerate(dataset_medians.index):
                y = model_data[model_data['dataset'] == dataset]['accuracy'] * 100
                x = np.random.normal(positions[j], width * 0.08, size=len(y))  # Reduced jitter from 0.1 to 0.08
                ax.scatter(x, y, alpha=0.6, s=16,
                           c=[self.color_scheme.get(model_name)],
                           zorder=2)

        # Customize the plot
        # ax.set_title('Model Performance Sensitivity Across Evaluation Dimensions',
        #              pad=25, fontsize=23, fontfamily='DejaVu Serif')
        ax.set_ylabel('Accuracy Score', fontsize=34, fontfamily='DejaVu Serif', labelpad=15)

        # Set axis properties with adjusted margins
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Adjust x-axis limits to reduce white space
        ax.set_xlim(-0.5, num_datasets - 0.5)  # This reduces the white space on sides

        # Customize ticks
        plt.xticks(range(num_datasets),
                   [self._wrap_labels(label) for label in dataset_medians.index],
                   rotation=45, ha='right', fontsize=32, fontfamily='DejaVu Serif')
        plt.yticks(fontsize=26, fontfamily='DejaVu Serif')

        # Adjust layout with tighter spacing
        plt.tight_layout()

        # Fine-tune the spacing
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.95)

        # Save plots
        plt.savefig('model_performance_mpl.png', dpi=300, bbox_inches='tight')
        plt.savefig('model_performance_mpl.pdf', bbox_inches='tight')

        print("Plots saved in PDF and PNG formats")

        # Close the figure to free memory
        plt.close()
def main():
    # Configuration
    base_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/ConfigurationOptimizer/data"

    # Define models
    models = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3'
    ]
    # models = ['sampled/sampled_data']

    # experiments = [ ZERO_SHOT_CONFIG]
    # Define datasets
    datasets = [
        "ai2_arc.arc_challenge",
        # "ai2_arc.arc_easy",
        "hellaswag",
        "openbook_qa",
        "social_iqa",
        "mmlu.college_biology",
        # "mmlu.high_school_government_and_politics",
        "mmlu_pro.law"
    ]
    # datasets = [
    #     # "ai2_arc.arc_challenge",
    #     # "ai2_arc.arc_easy",
    #     "hellaswag",
    #     # "openbook_qa",
    #     # "social_iqa",
    #     # "mmlu.college_biology",
    #     # "mmlu.high_school_government_and_politics",
    #     # "mmlu_pro.law"
    # ]

    # Process data
    processor = DataProcessor(base_dir, models, datasets)
    data = processor.load_and_process_data()

    # Create visualization
    visualizer = Visualizer()
    visualizer.save_legend_only(data)  # Save the legend separately
    visualizer.create_plot(data)


if __name__ == "__main__":
    main()