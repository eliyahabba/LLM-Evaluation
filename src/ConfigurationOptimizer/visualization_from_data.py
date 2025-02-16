import os
from typing import List, Dict, Tuple, Union, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

# Common configuration settings
METHOD_ORDER = {
    'Random Baseline': "#d62728",
    'Best Global Configuration': '#2ca02c',
    'Axis-wise Selection': '#1f77b4',
    'Regression Predictor': '#ff7f0e'
}

METHOD_DISPLAY_NAMES = {
    "Best Global Configuration": "Maximum observed prompt",
    "Axis-wise Selection": "Independent selection",
    "Regression Predictor": "Linear regression",
    "Random Baseline": "Random baseline"
}

PLOT_SETTINGS = {
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': lambda x: x + 9,
    'axes.titlesize': lambda x: x + 3,
    'legend.fontsize': lambda x: x
}


def scientific_formatter(x: float, pos: int) -> str:
    """Format numbers in scientific notation for x-axis."""
    return "0" if x == 0 else f"$10^{int(np.log10(x))}$"


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area under the curve using trapezoidal rule."""
    log_x = np.log10(x)
    norm_x = (log_x - log_x.min()) / (log_x.max() - log_x.min())
    return trapezoid(y / 100, norm_x)


def create_legend(methods: List[str], colors: List[str], linestyles: List[str],
                  fontsize: int = 12, ncol: int = 4) -> Tuple[plt.Figure, Any]:
    """Create a separate figure containing only the legend."""
    plt.figure(figsize=(12, 2))
    ax = plt.gca()

    legend_elements = []
    for method, color, ls in zip(methods, colors, linestyles):
        display_name = METHOD_DISPLAY_NAMES[method]
        line = plt.Line2D([0], [0], linestyle=ls, color=color,
                          label=f"{display_name} (Smoothed)", alpha=0.7, linewidth=8)
        scatter = plt.Line2D([0], [0], marker='o', color=color, linestyle='none',
                             label=f"{display_name} (Actual)", markersize=25)
        legend_elements.extend([line, scatter])

    legend = ax.legend(handles=legend_elements,
                       bbox_to_anchor=(0.5, 0.5),
                       loc='center',
                       ncol=ncol,
                       facecolor='white',
                       fontsize=fontsize,
                       frameon=True,
                       edgecolor='black',
                       fancybox=False)

    ax.set_facecolor('white')
    ax.set_axis_off()
    return plt.gcf(), legend


def plot_auc_comparison(auc_data: Dict[str, Dict[str, float]], output_file: str,
                        fontsize: int = 12) -> None:
    """Create a bar plot comparing AUC scores across models and methods."""
    plt.figure(figsize=(16, 8))

    # Apply plot settings
    for setting, value in PLOT_SETTINGS.items():
        plt.rcParams[setting] = value(fontsize) if callable(value) else value

    available_methods = [m for m in METHOD_ORDER.keys()
                         if m in set().union(*[d.keys() for d in auc_data.values()])]
    models = list(auc_data.keys())

    # Plot configuration
    bar_width = 0.15
    group_spacing = 0.8
    indices = np.arange(len(models)) * group_spacing

    # Create bars
    for i, method in enumerate(available_methods):
        values = [auc_data[model].get(method, 0) for model in models]
        positions = indices + (i - len(available_methods) / 2 + 0.5) * bar_width
        plt.bar(positions, values, bar_width, label=method,
                color=METHOD_ORDER[method], alpha=0.7)

    # Format model names
    formatted_models = []
    for model in models:
        if "Llama" in model or "Mistral" in model or "OLMoE" in model:
            formatted_models.append(model.replace("-Instruct", "\nInstruct"))
        else:
            formatted_models.append(model)

    # Customize plot
    plt.xticks(indices, formatted_models, rotation=0, ha='center', fontsize=fontsize + 8)
    plt.xlim(min(indices) - group_spacing / 2, max(indices) + group_spacing / 2)
    plt.ylabel('AUC (Lower is Better)')
    plt.yticks(fontsize=fontsize + 4)

    # Create and save legend separately
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_fig, legend_ax = plt.subplots(figsize=(22, 2))
    labels = [METHOD_DISPLAY_NAMES[label] for label in labels]

    legend = legend_ax.legend(handles, labels, loc='center', ncol=2,
                              frameon=True, edgecolor='black', fancybox=False,
                              facecolor='white', fontsize=fontsize * 1.75)

    legend_ax.set_facecolor('white')
    legend_ax.axis('off')

    # Save legend
    legend_output_file = os.path.splitext(output_file)[0] + '_legend.png'
    legend_fig.savefig(legend_output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(legend_fig)

    # Save main plot
    plt.tight_layout(pad=0.5)
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    plt.close()


def plot_combined_performance_gaps(model_names: List[str],
                                   input_dir: str = "output/plots",
                                   output_file: str = "combined_plots/combined_figure.png",
                                   smooth: bool = True,
                                   experiment_suffix: str = "_zero_shot",
                                   fontsize: int = 24) -> None:
    """Create a combined figure with performance gap plots for multiple models."""
    n_models = len(model_names)

    # Configure plot layout
    if n_models > 3:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        if n_models % 2 == 1:
            fig.delaxes(axes[5])
    else:
        fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
    axes = [axes] if n_models == 1 else axes

    # Setup common variables
    linestyles = ['-', '--', '-.', ':']
    all_auc_data = {}

    # Get methods and colors for legend
    first_model = model_names[0]
    csv_path = os.path.join(input_dir, f"{first_model.split('/')[-1]}{experiment_suffix}_smoothed.csv")
    df = pd.read_csv(csv_path)
    methods = [method for method in METHOD_ORDER.keys() if method in df['method'].unique()]
    colors = [METHOD_ORDER[method] for method in methods]

    # Create and save legend
    legend_fig, _ = create_legend(methods, colors, linestyles[:len(methods)], fontsize, ncol=4)
    legend_path = os.path.join(os.path.dirname(output_file), 'legend.png')
    legend_fig.savefig(legend_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(legend_fig)

    # Process each model
    for idx, (model_name, ax) in enumerate(zip(model_names, axes)):
        base_name = model_name.split('/')[-1]
        csv_path = os.path.join(input_dir, f"{base_name}{experiment_suffix}_smoothed.csv")

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Warning: Could not find data for {model_name} at {csv_path}")
            continue

        methods = [m for m in METHOD_ORDER.keys() if m in df['method'].unique()]
        colors = [METHOD_ORDER[m] for m in methods]
        all_auc_data[base_name] = {}

        # Plot each method
        for i, (method, color) in enumerate(zip(methods, colors)):
            method_data = df[df['method'] == method]
            x = np.array(method_data['sample_size'])[:-1]
            y = np.array(method_data['gap'])[:-1] * 100
            yerr = np.array(method_data['gap_std'])[:-1] * 100

            x_offset = x * (1 + (i - len(methods) / 2) * 0.02)
            all_auc_data[base_name][method] = compute_auc(x, y)

            if smooth and len(y) >= 3:
                window_length = min(len(y) - (len(y) + 1) % 2, 5)
                y_smooth = savgol_filter(y, window_length, 2)
                ax.plot(x_offset, y_smooth, linestyle=linestyles[i % len(linestyles)],
                        color=color, alpha=0.7)
                ax.scatter(x_offset, y, marker='o', color=color, s=30, zorder=5)
            else:
                ax.plot(x_offset, y, linestyle=linestyles[i % len(linestyles)], color=color)
                ax.scatter(x_offset, y, marker='o', color=color, s=30, zorder=5)

            # Add error bands
            ax.fill_between(x_offset,
                            np.maximum(y - yerr, 0),
                            y + yerr,
                            color=color,
                            alpha=0.2)

        # Configure subplot
        configure_subplot(ax, idx, base_name, fontsize)

    plt.tight_layout(pad=0.5, w_pad=0.5)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    # Create and save AUC comparison
    base_name, ext = os.path.splitext(output_file)
    plot_auc_comparison(all_auc_data, f"{base_name}_auc{ext}", fontsize)


def configure_subplot(ax: plt.Axes, idx: int, base_name: str, fontsize: int) -> None:
    """Configure individual subplot settings."""
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x)}%"))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylim(bottom=0, top=20)

    ax.grid(True, which='major', axis='both', linestyle='-', alpha=0.2)
    ax.set_axisbelow(True)

    # Configure ticks and labels
    ax.tick_params(axis='x', which='minor', length=0)
    if idx > 0 and idx != 3:
        ax.tick_params(axis='y', which='both', length=0, labelleft=False)

    # Set title and labels
    display_name = base_name.replace("Meta-Llama", "Llama")
    ax.set_title(display_name, fontsize=15)

    if idx == 1 or True:
        ax.set_xlabel('Number of Instances (Log Scale)', fontsize=14)
    if idx == 0 or idx == 3:
        ax.set_ylabel('Accuracy Gap vs. Best Prompt', fontsize=16)


if __name__ == "__main__":
    models_to_plot = [
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Llama-3.2-1B-Instruct',
    ]

    plot_combined_performance_gaps(
        model_names=models_to_plot,
        input_dir="output/plots",
        output_file="output/combined_plots/combined_figure.png",
        smooth=True,
        experiment_suffix="_zero_shot"
    )