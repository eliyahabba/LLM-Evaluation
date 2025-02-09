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


import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import os
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter


class Visualizer:
    """Handles plotting and results presentation."""

    def plot_performance_gaps(
            self,
            results: Dict,
            output_file: Optional[str] = None,
            model_name: Optional[str] = None,
            smooth: bool = True
    ):
        """Plot gap from optimum as function of sample size with improved error visualization."""
        plt.figure(figsize=(12, 7))

        colors = get_distinct_colors(len(results))
        linestyles = ['-', '--', '-.', ':']
        all_x_values = set()

        for i, ((method_name, method_results), color) in enumerate(zip(results.items(), colors)):
            x = np.array(method_results['sample_sizes'])
            y = np.array(method_results['gaps'])*100
            yerr = np.array(method_results['gap_stds'])*100

            all_x_values.update(x)
            x_offset = x * (1 + (i - len(results) / 2) * 0.02)

            if smooth:
                window_length = min(len(y) - (len(y) + 1) % 2, 5)
                if window_length >= 3:
                    y_smooth = savgol_filter(y, window_length, 2)
                    plt.plot(x_offset, y_smooth, linestyle=linestyles[i % len(linestyles)],
                             color=color, label=f"{method_name} (Smoothed)", alpha=0.7)
                    plt.scatter(x_offset, y, marker='o', color=color, s=30, label=f"{method_name} (Actual)", zorder=5)
                else:
                    plt.plot(x_offset, y, linestyle=linestyles[i % len(linestyles)], color=color, label=method_name)
                    plt.scatter(x_offset, y, marker='o', color=color, s=30, zorder=5)
            else:
                plt.plot(x_offset, y, linestyle=linestyles[i % len(linestyles)], color=color, label=method_name)
                plt.scatter(x_offset, y, marker='o', color=color, s=30, zorder=5)

            y_lower_bound = np.maximum(y - yerr, 0)
            y_upper_bound = y + yerr
            plt.fill_between(x_offset, y_lower_bound, y_upper_bound, color=color, alpha=0.2)

        # Convert x-values to sorted list (only actual data points)
        all_x_values = sorted(all_x_values)

        # Set log scale for x-axis
        plt.xscale('log')

        ax = plt.gca()

        # Only use actual data points as x-ticks
        ax.set_xticks(all_x_values)

        # Create labels, removing the second-to-last and replacing the last one with "Data Size"
        xtick_labels = [f"{int(x):,}" for x in all_x_values]

        if len(xtick_labels) > 1:
            xtick_labels[-2] = ""  # Remove second-to-last label
        xtick_labels[-1] = "Data Size"  # Set last label

        ax.set_xticklabels(xtick_labels, rotation=0, ha='center')

        # Rotate labels dynamically if needed
        if len(all_x_values) > 8:
            plt.xticks(rotation=45, ha='right')
        else:
            plt.xticks(rotation=0, ha='center')

        # Prevent Y from going below 0
        plt.ylim(bottom=0)

        plt.xlabel('Number of Samples')
        plt.ylabel('Score Gap from Optimal Configuration')
        plt.grid(True, which="both", ls="-", alpha=0.2)

        title = 'Performance Gap vs Sample Size'
        if model_name:
            title += f'\n{model_name}'
        plt.title(title)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        if output_file:
            base_name, ext = os.path.splitext(output_file)
            output_path = f"{base_name}_{'smoothed' if smooth else 'original'}{ext}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

            results_file = f"{base_name}_{'smoothed' if smooth else 'original'}.csv"
            with open(results_file, 'w') as f:
                f.write("method,sample_size,gap,gap_std\n")
                for method_name, method_results in results.items():
                    for size, gap, std in zip(
                            method_results['sample_sizes'],
                            method_results['gaps'],
                            method_results['gap_stds']
                    ):
                        f.write(f"{method_name},{size},{gap},{std}\n")
        else:
            plt.show()