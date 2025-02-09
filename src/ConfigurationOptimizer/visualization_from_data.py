import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def get_distinct_colors(n):
    """Return a list of n visually distinct, high-contrast colors."""
    color_palette = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"  # Cyan
    ]
    return color_palette[:n] if n <= len(color_palette) else plt.cm.tab10(np.linspace(0, 1, n))


def plot_combined_performance_gaps(
        model_names,
        input_dir="output/plots",
        output_file="combined_plots.png",
        smooth=True,
        experiment_suffix="_zero_shot",
        fontsize=10
):
    """
    Create a combined figure with performance gap plots for multiple models.

    Args:
        model_names (list): List of model names to include in the plot
        input_dir (str): Directory containing the CSV files
        output_file (str): Output file path for the combined figure
        smooth (bool): Whether to apply smoothing to the curves
        experiment_suffix (str): Suffix for the experiment files
    """
    n_models = len(model_names)
    # Use LaTeX-style font for better paper quality
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    # Set global font sizes
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize + 2
    plt.rcParams['axes.titlesize'] = fontsize + 2
    plt.rcParams['legend.fontsize'] = fontsize

    # If only one model, make axes iterable
    if n_models == 1:
        axes = [axes]

    linestyles = ['-', '--', '-.', ':']

    # Process each model
    for idx, (model_name, ax) in enumerate(zip(model_names, axes)):
        # Read the CSV file
        base_name = model_name.split('/')[-1]
        csv_path = os.path.join(input_dir, f"{base_name}{experiment_suffix}_smoothed.csv")
        df = pd.read_csv(csv_path)

        # Get unique methods and sample sizes
        methods = df['method'].unique()
        colors = get_distinct_colors(len(methods))

        all_x_values = set()

        # Plot each method
        for i, (method, color) in enumerate(zip(methods, colors)):
            method_data = df[df['method'] == method]
            x = np.array(method_data['sample_size'])
            y = np.array(method_data['gap']) * 100
            yerr = np.array(method_data['gap_std']) * 100

            all_x_values.update(x)
            x_offset = x * (1 + (i - len(methods) / 2) * 0.02)

            if smooth:
                window_length = min(len(y) - (len(y) + 1) % 2, 5)
                if window_length >= 3:
                    y_smooth = savgol_filter(y, window_length, 2)
                    ax.plot(x_offset, y_smooth, linestyle=linestyles[i % len(linestyles)],
                            color=color, label=f"{method} (Smoothed)" if idx == n_models - 1 else None, alpha=0.7)
                    ax.scatter(x_offset, y, marker='o', color=color, s=30,
                               label=f"{method} (Actual)" if idx == n_models - 1 else None, zorder=5)
                else:
                    ax.plot(x_offset, y, linestyle=linestyles[i % len(linestyles)],
                            color=color, label=method if idx == n_models - 1 else None)
                    ax.scatter(x_offset, y, marker='o', color=color, s=30, zorder=5)
            else:
                ax.plot(x_offset, y, linestyle=linestyles[i % len(linestyles)],
                        color=color, label=method if idx == n_models - 1 else None)
                ax.scatter(x_offset, y, marker='o', color=color, s=30, zorder=5)

            y_lower_bound = np.maximum(y - yerr, 0)
            y_upper_bound = y + yerr
            ax.fill_between(x_offset, y_lower_bound, y_upper_bound, color=color, alpha=0.2)

        # Configure each subplot
        ax.set_xscale('log')
        all_x_values = sorted(all_x_values)
        ax.set_xticks(all_x_values)

        xtick_labels = [f"{int(x):,}" for x in all_x_values]
        if len(xtick_labels) > 1:
            xtick_labels[-2] = ""
        xtick_labels[-1] = "Data Size"

        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
        ax.set_ylim(bottom=0)
        # Add percentage signs to y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}\%'.format(y)))

        # Improved grid with major/minor lines
        ax.grid(True, which="major", ls="-", alpha=0.2)
        ax.grid(True, which="minor", ls=":", alpha=0.1)

        # Set titles and labels
        # Format model name for title (remove organization prefix)
        display_name = base_name.split('-Instruct')[0]  # Remove '-Instruct' suffix if present
        ax.set_title(display_name)
        ax.set_xlabel('Number of Samples')
        if idx == 0:  # Only add y-label to leftmost subplot
            ax.set_ylabel('Performance Gap (\%)')

        # Add legend only to rightmost subplot
        if idx == n_models - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # Example usage
    models_to_plot = [
        'mistralai/Mistral-7B-Instruct-v0.3',
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct'
    ]

    plot_combined_performance_gaps(
        model_names=models_to_plot,
        input_dir="output/plots",
        output_file="output/combined_plots/combined_figure.png",
        smooth=True,
        experiment_suffix="_zero_shot"
    )