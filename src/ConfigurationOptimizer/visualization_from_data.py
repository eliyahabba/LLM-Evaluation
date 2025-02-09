# visualization_combined.py
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import integrate
from scipy.signal import savgol_filter

method_order = {
    'Random Baseline':  "#d62728",
    'Best Global Configuration': '#2ca02c',
    'Axis-wise Optimization': '#1f77b4',
    'Regression Predictor': '#ff7f0e'
}


def get_distinct_colors(n: int) -> List:
    """Return a list of n visually distinct colors."""
    color_palette = [
        "#2ca02c",  # Green
        "#1f77b4",  # Blue
        "#d62728",  # Red
        "#ff7f0e",  # Orange
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"  # Cyan
    ]
    return color_palette[:n] if n <= len(color_palette) else plt.cm.tab10(np.linspace(0, 1, n))


def scientific_formatter(x: float, pos: int) -> str:
    """Format numbers in scientific notation for x-axis."""
    if x == 0:
        return "0"
    exp = int(np.log10(x))
    return f"$10^{exp}$"


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute area under the curve (AUC) using trapezoidal rule.
    Lower AUC indicates better performance.
    """
    log_x = np.log10(x)
    norm_x = (log_x - log_x.min()) / (log_x.max() - log_x.min())
    norm_y = y / 100  # Since y is in percentages
    return integrate.trapz(norm_y, norm_x)


def plot_auc_comparison(auc_data: dict, output_file: str, fontsize: int = 12) -> None:
    """Create a bar plot comparing AUC scores across models and methods."""
    # Define the custom order for methods
    plt.figure(figsize=(5, 5))  # Reduced size to fit one column

    # Configure plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize + 2
    plt.rcParams['axes.titlesize'] = fontsize + 2
    plt.rcParams['legend.fontsize'] = fontsize

    # Get all available methods
    available_methods = set().union(*[d.keys() for d in auc_data.values()])

    # Filter and order methods according to the defined order
    methods = [m for m in available_methods]
    models = list(auc_data.keys())
    # sort the tuples by the method order
    # Set bar positions with thinner bars
    bar_width = 0.15  # Fixed smaller width for bars

    # Plot bars for each method
    for i, (method, color) in enumerate(method_order.items()):
        positions = np.arange(len(models)) + i * bar_width - (len(methods) - 1) * bar_width / 2
        values = [auc_data[model].get(method, 0) for model in models]
        plt.bar(positions, values, bar_width, label=method, color=color, alpha=0.7)

    # Customize plot
    plt.xticks(np.arange(len(models)),
               [name.split('-Instruct')[0] for name in models],
               rotation=45, ha='right')
    plt.ylabel('AUC (lower is better)')
    plt.title('AUC Comparison Across Models')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()


def plot_combined_performance_gaps(
        model_names: List[str],
        input_dir: str = "output/plots",
        output_file: str = "combined_plots.png",
        smooth: bool = True,
        experiment_suffix: str = "_zero_shot",
        fontsize: int = 12,
        show_legend: bool = True,
) -> None:
    """
    Create a combined figure with performance gap plots for multiple models.
    """
    n_models = len(model_names)

    # Configure plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize + 2
    plt.rcParams['axes.titlesize'] = fontsize + 2
    plt.rcParams['legend.fontsize'] = fontsize

    # Create subplot layout with reduced width
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

    # Make axes iterable if only one model
    if n_models == 1:
        axes = [axes]

    linestyles = ['-', '--', '-.', ':']
    all_auc_data = {}  # Store AUC data for each model and method

    # Process each model
    for idx, (model_name, ax) in enumerate(zip(model_names, axes)):
        base_name = model_name.split('/')[-1]
        csv_path = os.path.join(input_dir, f"{base_name}{experiment_suffix}_smoothed.csv")

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Warning: Could not find data for {model_name} at {csv_path}")
            continue

        # Get unique methods but sort them according to method_order
        methods = [method for method in method_order.keys() if method in df['method'].unique()]
        colors = [method_order[method] for method in methods]
        # Get unique methods and assign colors
        # methods = df['method'].unique()
        # colors = get_distinct_colors(len(methods))
        all_auc_data[base_name] = {}

        # Plot each method
        for i, (method, color) in enumerate(zip(methods, colors)):
            method_data = df[df['method'] == method]
            x = np.array(method_data['sample_size'])
            y = np.array(method_data['gap']) * 100
            yerr = np.array(method_data['gap_std']) * 100

            # Calculate AUC
            auc = compute_auc(x, y)
            all_auc_data[base_name][method] = auc

            x_offset = x * (1 + (i - len(methods) / 2) * 0.02)

            if smooth and len(y) >= 3:
                window_length = min(len(y) - (len(y) + 1) % 2, 5)
                y_smooth = savgol_filter(y, window_length, 2)
                ax.plot(x_offset, y_smooth,
                        linestyle=linestyles[i % len(linestyles)],
                        color=color,
                        label=f"{method} (Smoothed)" if show_legend and idx == n_models - 1 else None,
                        alpha=0.7)
                ax.scatter(x_offset, y,
                           marker='o',
                           color=color,
                           s=30,
                           label=f"{method} (Actual)" if show_legend and idx == n_models - 1 else None,
                           zorder=5)
            else:
                ax.plot(x_offset, y,
                        linestyle=linestyles[i % len(linestyles)],
                        color=color,
                        label=method if show_legend and idx == n_models - 1 else None)
                ax.scatter(x_offset, y, marker='o', color=color, s=30, zorder=5)

            # Add error bands
            y_lower_bound = np.maximum(y - yerr, 0)
            y_upper_bound = y + yerr
            ax.fill_between(x_offset, y_lower_bound, y_upper_bound,
                            color=color, alpha=0.2)

        # Configure subplot
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x)}%"))
        ax.set_ylim(bottom=0)

        # Add grid
        ax.grid(True, which="major", ls="-", alpha=0.2)
        ax.grid(True, which="minor", ls=":", alpha=0.1)

        # Set titles and labels
        display_name = base_name.split('-Instruct')[0]
        ax.set_title(display_name)
        ax.set_xlabel('Number of Samples (Log Scale)')
        if idx == 0:  # Only add y-label to leftmost subplot
            ax.set_ylabel('Accuracy Gap vs. Best Configuration')

        # Add legend to rightmost subplot only if show_legend is True
        if show_legend and idx == n_models - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Add padding to prevent y-label from being cut off
    plt.tight_layout(pad=2.0, w_pad=1.0)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save main plot
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    # Create and save AUC comparison plot
    base_name, ext = os.path.splitext(output_file)
    plot_auc_comparison(all_auc_data, f"{base_name}_auc{ext}", fontsize)


if __name__ == "__main__":
    # Example usage
    models_to_plot = [
        'mistralai/Mistral-7B-Instruct-v0.3',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct'
    ]
    # Model configurations
    models_to_plot = [
        'meta-llama/Llama-3.2-1B-Instruct',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]

    # Create version with legend and AUC plot
    plot_combined_performance_gaps(
        model_names=models_to_plot,
        input_dir="output/plots",
        output_file="output/combined_plots/combined_figure_with_legend.png",
        smooth=True,
        experiment_suffix="_zero_shot",
        show_legend=True
    )

    # Create version without legend or AUC plot
    plot_combined_performance_gaps(
        model_names=models_to_plot,
        input_dir="output/plots",
        output_file="output/combined_plots/combined_figure_no_legend.png",
        smooth=True,
        experiment_suffix="_zero_shot",
        show_legend=False
    )
