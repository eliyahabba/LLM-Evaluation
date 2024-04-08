import pandas as pd
import streamlit as st
from src.Analysis.McNemarTestFromTable import McNemarTestFromTable
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class PerformAnalysis:
    def __init__(self, grouped_metadata_df: pd.DataFrame, best_row: pd.Series):
        self.grouped_metadata_df = grouped_metadata_df
        self.best_row = best_row

    def calculate_mcnemar_test(self) -> None:
        """
        Calculates the McNemar test for the given model and dataset.
        """
        df = McNemarTestFromTable.perform_mcnemar_test_from_table(self.grouped_metadata_df)
        with st.expander("McNemar Test Results"):
            st.write(df)

    def calculate_cochrans_q_test(self) -> None:
        """
        Calculates the Cochran's Q test for the given model and dataset.
        @return:
        """
        result = McNemarTestFromTable.perform_cochrans_q_test_from_table(self.grouped_metadata_df)
        with st.expander("Cochran's Q Test Results"):
            st.write("Cochran's Q Test Statistic:", result.statistic)
            st.write(f"Cochran's Q Test P-value:{result.pvalue}")
