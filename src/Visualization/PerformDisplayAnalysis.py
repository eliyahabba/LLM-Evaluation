from pathlib import Path

import pandas as pd
import streamlit as st

from src.Visualization.PerformAnalysis import PerformAnalysis
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class PerformDisplayAnalysis:
    def __init__(self, comparison_matrix_path: Path, grouped_metadata_df: pd.DataFrame, best_row: pd.Series):
        self.comparison_matrix_path = comparison_matrix_path
        self.performance_summary_df = pd.read_csv(self.comparison_matrix_path)
        self.grouped_metadata_df = grouped_metadata_df
        self.best_row = best_row

    def display_mcnemar_test(self, best_row: pd.Series) -> None:
        """
        Calculates the McNemar test for the given model and dataset.
        """
        perform_analysis = PerformAnalysis(self.comparison_matrix_path, self.grouped_metadata_df, self.best_row)
        result = perform_analysis.calculate_mcnemar_test(best_row)
        with st.expander("McNemar Test Results"):
            st.markdown("The row / column that corresponds to the best template is marked with 'best'.")
            st.write(result)

    def display_cochrans_q_test(self) -> None:
        """
        Calculates the Cochran's Q test for the given model and dataset.
        @return:
        """
        perform_analysis = PerformAnalysis(self.comparison_matrix_path, self.grouped_metadata_df, self.best_row)
        result = perform_analysis.calculate_cochrans_q_test()
        with st.expander("Cochran's Q Test Results"):
            st.write("Statistic:", f"{result.statistic:.2f}")
            st.write(f"P-value:{result.pvalue:.2f}")
