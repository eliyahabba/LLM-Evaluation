#!/usr/bin/env python3
"""
Entry point for robustness analysis.
Runs the RobustnessAnalyzer to create per-question robustness histograms.
"""

from src.analysis.plotting.plotters. robustness_analysis import main

if __name__ == "__main__":
    main() 