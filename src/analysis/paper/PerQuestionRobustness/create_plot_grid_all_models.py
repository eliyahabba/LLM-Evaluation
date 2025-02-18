import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot configuration constants
PLOT_SETTINGS = {
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif'
}

FONT_SIZES = {
    'xlabel': 24,
    'ylabel': 22,
    'title': 21,
    'ticks': 19,
    'bar_labels': 12
}

BAR_COLORS = {
    'high': 'green',  # >= 90%
    'low': 'red',  # <= 10%
    'medium': 'skyblue'  # Everything else
}


def get_df(model_path: str) -> pd.DataFrame:
    """Load and filter parquet files from the model directory."""
    dfs = []
    for file in os.listdir(model_path):
        file_path = os.path.join(model_path, file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
    df = pd.concat(dfs)
    return df[df['total_configurations'] > 3000]


def color_bars(patches: List, bins: np.ndarray) -> None:
    """Color histogram bars based on their bin values."""
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i + 1]) / 2
        if bin_center >= 90:
            patch.set_facecolor(BAR_COLORS['high'])
        elif bin_center <= 10:
            patch.set_facecolor(BAR_COLORS['low'])
        else:
            patch.set_facecolor(BAR_COLORS['medium'])


def add_bar_labels(ax: plt.Axes, patches: List, counts: np.ndarray) -> None:
    """Add count labels above each bar."""
    for patch, count in zip(patches, counts):
        ax.text(patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                f'{int(count)}',
                ha='center',
                va='bottom',
                color='darkblue',
                fontsize=FONT_SIZES['bar_labels'])


def configure_axes(ax: plt.Axes, is_bottom: bool, is_left: bool, model_name: str) -> None:
    """Configure axes labels, ticks, and titles."""
    # X-axis configuration
    if is_bottom:
        ax.set_xlabel('Success Rate Over Different Prompts (%)',
                      fontsize=FONT_SIZES['xlabel'],
                      labelpad=17)
    else:
        ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=FONT_SIZES['ticks'])
    ax.set_xlim(0, 100)

    # Y-axis configuration
    if is_left:
        ax.set_ylabel('Number of Instances',
                      fontsize=FONT_SIZES['ylabel'])
        ax.tick_params(axis='y', labelsize=FONT_SIZES['ticks'])
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    # Grid and title
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title(model_name.replace('Meta-Llama', 'Llama'),
                 fontsize=FONT_SIZES['title'])


def plot_histogram_subplot(df: pd.DataFrame, ax: plt.Axes, model_name: str,
                           is_bottom: bool = False, is_left: bool = False) -> None:
    """Create a histogram subplot for a single model."""
    percentage_correct = df['fraction_correct'] * 100
    bins = np.arange(0, 105, 5)  # Include 100 in the last bin

    # Create histogram
    counts, bins, patches = ax.hist(percentage_correct,
                                    bins=bins,
                                    edgecolor='black',
                                    color='skyblue',
                                    alpha=0.7,
                                    rwidth=1)

    color_bars(patches, bins)
    add_bar_labels(ax, patches, counts)
    configure_axes(ax, is_bottom, is_left, model_name)


def create_grid_plot(models_to_evaluate: List[str], base_path: str) -> None:
    """Create a grid of histograms for the specified models."""
    # Apply plot settings
    for setting, value in PLOT_SETTINGS.items():
        plt.rcParams[setting] = value

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    axes[-1].axis('off')  # Disable last subplot

    # Plot each model
    for i, model in enumerate(models_to_evaluate):
        if i < len(axes) - 1:  # Ensure we don't exceed available subplots
            col = i % 6
            model_name = model.replace("/", "_")
            model_path = os.path.join(base_path, model_name)
            df = get_df(model_path)

            is_center = (col == 4)
            is_left = (col == 0) or (col == 3)

            plot_histogram_subplot(df,
                                   axes[col],
                                   model_name.split('_')[-1],
                                   is_bottom=is_center,
                                   is_left=is_left)

            print(f"Total samples in dataset for {model_name}: {len(df)}")

    plt.tight_layout()
    plt.savefig('model_histograms_grid2_3.png', dpi=300)
    plt.savefig('model_histograms_grid2_3.pdf', bbox_inches='tight')
    plt.close()


def main():
    """Run the main analysis pipeline for creating a grid of histograms."""
    models_to_evaluate = [
        'mistralai/Mistral-7B-Instruct-v0.3',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
    ]
    base_path = r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/app/results_local/Shots/'
    create_grid_plot(models_to_evaluate, base_path)


if __name__ == "__main__":
    main()