from pathlib import Path
from typing import Union

import pandas as pd

from src.Analysis.StatisticalTests.CompareSeriesBinaryDataFromTable import CompareSeriesBinaryDataFromTable
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class PerformAnalysis:
    def __init__(self, comparison_matrix_path: Path, grouped_metadata_df: pd.DataFrame, best_row: pd.Series):
        self.comparison_matrix_path = comparison_matrix_path
        self.performance_summary_df = pd.read_csv(self.comparison_matrix_path)
        self.grouped_metadata_df = grouped_metadata_df
        self.best_row = best_row

    def group_performance_summary_by_template(self, top_k: Union[int, None]) -> pd.DataFrame:
        """
        Groups the performance summary by the template name.
        @param top_k: The number of top results to compare.
        """
        # take the columns from self.performance_summary_df and group them by the rows in
        # templater column of self.grouped_metadata_df
        best_k_groups = self.grouped_metadata_df['accuracy'].nlargest(top_k).index if top_k else (
            self.grouped_metadata_df)['accuracy'].index
        best_grouped_metadata_df = self.grouped_metadata_df.loc[best_k_groups]
        groups = best_grouped_metadata_df['template_name'].values.tolist()
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
        else:
            # flatten the list of lists
            groups = [item for sublist in groups for item in sublist]
            performance_summary_df = performance_summary_df[groups]
        return performance_summary_df

    def process_data_for_cochrans_q_test(self, top_k: Union[int, None]) -> pd.DataFrame:
        """
        Processes the data for the Cochran's Q test.
        @param top_k: The number of top results to compare.
        @return: The processed data.
        """
        return self.group_performance_summary_by_template(top_k)

    def process_data_for_mcnemar_test(self, top_k: int) -> pd.DataFrame:
        """
        Processes the data for the McNemar test.
        @param top_k: The number of top results to compare.
        @return: The processed data.
        """
        return self.group_performance_summary_by_template(top_k)

    def calculate_mcnemar_test(self, best_row: pd.Series, top_k: int) -> pd.DataFrame:
        """
        Calculates the McNemar test for the given model and dataset.

        @param best_row: The best row.
        @param top_k: The number of top results to compare.

        @return: The result of the McNemar test.
        """
        mcnemar_df = self.process_data_for_mcnemar_test(top_k)
        result = CompareSeriesBinaryDataFromTable.perform_mcnemar_test_from_table(mcnemar_df)
        if 'template_name' in best_row:
            best_templates_numbers = ",".join([(x.split('template_')[1]) for x in best_row.template_name])
            best_templates_name = \
                f"{'templates ' if len(best_row.template_name) > 1 else 'template_'}{best_templates_numbers}"

            # add "best" to the row and column that corresponds to the best template
            result.index = result.index.map(lambda x: f"best set: {x}" if x == best_templates_name else x)
            result.columns = result.columns.map(lambda x: f"best set: {x}" if x == best_templates_name else x)
        return result

    def calculate_cochrans_q_test(self, top_k: Union[int, None]) -> pd.DataFrame:
        """
        Calculates the Cochran's Q test for the given model and dataset.
        @top_k: The number of top results to compare.

        @return: The result of the Cochran's Q test.
        """
        cochrans_q_df = self.process_data_for_cochrans_q_test(top_k)
        result = CompareSeriesBinaryDataFromTable.perform_cochrans_q_test_from_table(cochrans_q_df)
        return result
