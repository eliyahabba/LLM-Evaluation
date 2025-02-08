import os
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
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
        "#17becf"   # Cyan
    ]
    return color_palette[:n] if n <= len(color_palette) else plt.cm.tab10(np.linspace(0, 1, n))


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
        linestyles = ['-', '--', '-.', ':']  # Different line styles to differentiate methods

        for i, ((method_name, method_results), color) in enumerate(zip(results.items(), colors)):
            x = np.array(method_results['sample_sizes'])
            y = np.array(method_results['gaps'])
            yerr = np.array(method_results['gap_stds'])

            # Offset X slightly to avoid overlap in error bars
            x_offset = x * (1 + (i - len(results) / 2) * 0.02)  # Small shift for each method

            if smooth:
                # Apply Savitzky-Golay filter for smoothing
                window_length = min(len(y) - (len(y) + 1) % 2, 5)  # Must be odd and less than data length
                if window_length >= 3:
                    y_smooth = savgol_filter(y, window_length, 2)
                    plt.plot(x_offset, y_smooth, linestyle=linestyles[i % len(linestyles)],
                             color=color, label=f"{method_name} (Smoothed)")
                else:
                    plt.plot(x_offset, y, linestyle=linestyles[i % len(linestyles)],
                             color=color, label=method_name)

            # Use `fill_between` for better visibility of error margins
            plt.fill_between(x_offset, y - yerr, y + yerr, color=color, alpha=0.2)

            # Add label "Data Size" at the last data point
            last_x = x_offset[-1]
            last_y = y[-1]
            plt.text(last_x, last_y, "Data Size", fontsize=12, ha='right', va='bottom', color=color)

        plt.xlabel('Number of Samples')
        plt.ylabel('Score Gap from Optimal Configuration')
        plt.ticklabel_format(style='plain', axis='x')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)

        title = 'Performance Gap vs Sample Size'
        if model_name:
            title += f'\n{model_name}'
        plt.title(title)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if output_file:
            # Save both smoothed and original versions
            base_name, ext = os.path.splitext(output_file)
            output_path = f"{base_name}_{'smoothed' if smooth else 'original'}{ext}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Save the raw data
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