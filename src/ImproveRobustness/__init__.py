"""
Prompt Dimension Robustness Experiment Package.
This package contains modules for testing whether exposure to prompt variations
improves model robustness to both unseen dimension values and new datasets.
"""

from src.ImproveRobustness.main import run_experiment

__all__ = ['run_experiment']
