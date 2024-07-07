from pathlib import Path

import pandas as pd
import streamlit as st

from src.Analysis.PerformAnalysis import PerformAnalysis
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class AnalysisDisplay:
    def __init__(self, comparison_matrix_path: Path, grouped_metadata_df: pd.DataFrame, best_row: pd.Series,
                 ask_on_key:bool=True,default_top_k: int = 5):
        self.comparison_matrix_path = comparison_matrix_path
        self.performance_summary_df = pd.read_csv(self.comparison_matrix_path)
        self.grouped_metadata_df = grouped_metadata_df
        self.best_row = best_row
        if ask_on_key:
            self.top_k = self.select_top_k(default_top_k)
        else:
            self.top_k = default_top_k

    def select_top_k(self, default_top_k: int) -> int:
        """
        Select the top K templates to compare.
        @param default_top_k: The default top K value.
        @return: The top K templates to compare.
        """
        top_k = st.number_input("Select the top K templates (by accuracy) to compare", min_value=1, value=default_top_k)
        return top_k

    def get_results_of_mcnemar_test(self, best_row: pd.Series) -> pd.DataFrame:
        """
        calculate the McNemar test for the given model and dataset.
        @param best_row: The row to calculate the McNemar test with all other rows.
        @return: The results of the McNemar test.
        """
        perform_analysis = PerformAnalysis(self.comparison_matrix_path, self.grouped_metadata_df, self.best_row)
        result = perform_analysis.calculate_mcnemar_test(best_row, self.top_k)
        return result

    def display_mcnemar_test(self, best_row: pd.Series) -> None:
        """
        Calculates the McNemar test for the given model and dataset.
        @param best_row: The best row.
        @return: None
        """
        result = self.get_results_of_mcnemar_test(best_row)
        with st.expander("McNemar Test Results"):
            st.markdown("The row / column that corresponds to the best template is marked with 'best'.")
            st.write(result)

    def get_results_of_cochrans_q_test(self) -> pd.DataFrame:
        """
        Calculates the Cochran's Q test for the given model and dataset.
        @return: The results of the Cochran's Q test.
        """
        perform_analysis = PerformAnalysis(self.comparison_matrix_path, self.grouped_metadata_df, self.best_row)
        result = perform_analysis.calculate_cochrans_q_test(self.top_k)
        return result

    def display_cochrans_q_test(self) -> None:
        """
        Calculates the Cochran's Q test for the given model and dataset.
        @return: None
        """
        result = self.get_results_of_cochrans_q_test()
        with st.expander("Cochran's Q Test Results"):
            st.write("Statistic:", f"{result.statistic:.2f}")
            st.write(f"P-value:{result.pvalue:.2f}")
