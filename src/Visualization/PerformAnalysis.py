from pathlib import Path

import pandas as pd
import streamlit as st

from src.Analysis.StatisticalTests.CompareSeriesBinaryDataFromTable import CompareSeriesBinaryDataFromTable
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class PerformAnalysis:
    def __init__(self, performance_summary_path: Path, grouped_metadata_df: pd.DataFrame, best_row: pd.Series):
        self.performance_summary_path = performance_summary_path
        self.performance_summary_df = pd.read_csv(self.performance_summary_path)
        self.grouped_metadata_df = grouped_metadata_df
        self.best_row = best_row

    def group_performance_summary_by_template(self) -> pd.DataFrame:
        """
        Groups the performance summary by the template name.
        """
        # take the columns from self.performance_summary_df and group them by the rows in
        # templater column of self.grouped_metadata_df
        groups = self.grouped_metadata_df['template_name'].values.tolist()
        # The columns in the performance_summary_df are in the format 'experiment_1', 'experiment_2', etc, so we need to
        # extract the number 'template_name' from the 'experiment_' string,
        # to be with same format as the performance_summary_df
        new_columns = [x.split('experiment_')[1] for x in self.performance_summary_df.columns]

        performance_summary_df = self.performance_summary_df.copy()
        performance_summary_df.columns = new_columns
        # Merge columns in each group
        if any([len(group) > 1 for group in groups]):
            for group in groups:
                # Create a new column by finding the mode in each row
                templates_numbers = ",".join([(x.split('template_')[1]) for x in group])
                new_column_name = f"templates {templates_numbers}"
                performance_summary_df[new_column_name] = performance_summary_df[group].mode(axis=1).max(axis=1)
            performance_summary_df.drop(columns=sum(groups, []), inplace=True)
        return performance_summary_df

    def process_data_for_cochrans_q_test(self) -> pd.DataFrame:
        return self.group_performance_summary_by_template()

    def process_data_for_mcnemar_test(self) -> pd.DataFrame:
        return self.group_performance_summary_by_template()

    def calculate_mcnemar_test(self, best_row: pd.Series) -> None:
        """
        Calculates the McNemar test for the given model and dataset.
        """
        mcnemar_df = self.process_data_for_mcnemar_test()
        result = CompareSeriesBinaryDataFromTable.perform_mcnemar_test_from_table(mcnemar_df)
        best_templates_numbers = ",".join([(x.split('template_')[1]) for x in best_row.template_name])
        best_templates_name = f"{'templates ' if len(best_row.template_name)>1 else 'template_'}{best_templates_numbers}"

        # add "best" to the row and column that corresponds to the best template
        result.index = result.index.map(lambda x: f"best set: {x}" if x == best_templates_name else x)
        result.columns = result.columns.map(lambda x: f"best set {x}" if x == best_templates_name else x)
        with st.expander("McNemar Test Results"):
            st.markdown("The row / column that corresponds to the best template is marked with 'best'.")
            st.write(result)

    def calculate_cochrans_q_test(self) -> None:
        """
        Calculates the Cochran's Q test for the given model and dataset.
        @return:
        """
        cochrans_q_df = self.process_data_for_cochrans_q_test()
        result = CompareSeriesBinaryDataFromTable.perform_cochrans_q_test_from_table(cochrans_q_df)

        with st.expander("Cochran's Q Test Results"):
            st.write("Statistic:", f"{result.statistic:.2f}")
            st.write(f"P-value:{result.pvalue:.2f}")
