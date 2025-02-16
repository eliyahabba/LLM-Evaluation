import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import integrate
from scipy.signal import savgol_filter


def get_distinct_colors(n):
    """Return a list of n visually distinct, high-contrast colors."""
    color_palette = [
        "#2ca02c",  # Green
        "#1f77b4",  # Blue
        "#d62728",  # Red
        "#ff7f0e",  # Orange
        "#9467bd",  # Purple
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"  # Cyan
    ]
    return color_palette[:n] if n <= len(color_palette) else plt.cm.tab10(np.linspace(0, 1, n))


class Visualizer:
    """Handles plotting and results presentation."""

    def compute_auc(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute area under the curve (AUC) using trapezoidal rule.
        Lower AUC indicates better performance.
        """
        # Convert to log space for x axis
        log_x = np.log10(x)
        # Normalize x to [0,1] range for comparable AUC across different x ranges
        norm_x = (log_x - log_x.min()) / (log_x.max() - log_x.min())
        # Normalize y to [0,1] range
        norm_y = y / 100  # Since y is in percentages
        return integrate.trapz(norm_y, norm_x)

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

        # Calculate AUC for each method
        auc_scores = {}

        for i, ((method_name, method_results), color) in enumerate(zip(results.items(), colors)):
            x = np.array(method_results['sample_sizes'])
            y = np.array(method_results['gaps']) * 100
            yerr = np.array(method_results['gap_stds']) * 100

            # Calculate AUC
            auc_scores[method_name] = self.compute_auc(x, y)

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

        # Set log scale for x-axis
        plt.xscale('log')

        ax = plt.gca()

        # Format x-axis with scientific notation
        def scientific_formatter(x, pos):
            if x == 0:
                return "0"
            exp = int(np.log10(x))
            return f"$10^{exp}$"

        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))

        # Reduce number of ticks
        ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=5))

        # Fix percentage symbol in y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x)}%"))

        # Prevent Y from going below 0
        plt.ylim(bottom=0)

        plt.xlabel('Number of Samples (Log Scale)')
        plt.ylabel('Accuracy Drop Compared to Best Configuration')

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

            # Save results including AUC scores
            results_file = f"{base_name}_{'smoothed' if smooth else 'original'}.csv"
            with open(results_file, 'w') as f:
                f.write("method,sample_size,gap,gap_std,auc\n")
                for method_name, method_results in results.items():
                    for size, gap, std in zip(
                            method_results['sample_sizes'],
                            method_results['gaps'],
                            method_results['gap_stds']
                    ):
                        f.write(f"{method_name},{size},{gap},{std},{auc_scores[method_name]}\n")
        else:
            plt.show()
