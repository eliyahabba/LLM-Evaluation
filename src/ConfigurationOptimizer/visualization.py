# visualization.py
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt


class Visualizer:
    """Handles plotting and results presentation."""

    def plot_performance_gaps(
            self,
            results: Dict,
            output_file: Optional[str] = None,
            model_name: Optional[str] = None
    ):
        """Plot gap from optimum as function of sample size."""
        plt.figure(figsize=(10, 6))

        for method_name, method_results in results.items():
            plt.plot(
                method_results['sample_sizes'],
                method_results['gaps'],
                marker='o',
                label=method_name
            )

        plt.xlabel('Num of Samples')
        plt.ylabel('Performance Gap')
        plt.ticklabel_format(style='plain', axis='x')
        plt.xscale('log')
        title = 'Performance Gap vs Sample Size'
        if model_name:
            title += f'\n{model_name}'
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
