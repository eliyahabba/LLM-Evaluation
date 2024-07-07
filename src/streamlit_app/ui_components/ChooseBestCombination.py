from pathlib import Path
from typing import Tuple

import pandas as pd

from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants


class ChooseBestCombination:
    def __init__(self, dataset_file_name: str, performance_summary_path: Path, selected_best_value_axes: list):
        self.dataset_file_name = dataset_file_name
        self.performance_summary_path = performance_summary_path
        self.selected_best_value_axes = selected_best_value_axes

    def choose_best_combination(self, is_choose_across_axes: bool = False, exclude_templates: list = None,
                                ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Choose the best combination of the values of the axes.
        @return: The best combination of the values of the axes.
        """
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / self.dataset_file_name / TemplatesGeneratorConstants.TEMPLATES_METADATA
        metadata_df = pd.read_csv(metadata_file, index_col=ResultConstants.TEMPLATE_NAME)
        grouped_metadata_df = self.get_grouped_metadata_df(metadata_df=metadata_df, exclude_templates=exclude_templates)
        if is_choose_across_axes:
            best_row = self.choose_across_axes(grouped_metadata_df)
        else:
            best_row = self.get_best_row(grouped_metadata_df)
        return grouped_metadata_df, best_row

    def get_grouped_metadata_df(self, metadata_df: pd.DataFrame, exclude_templates: list = None) -> pd.DataFrame:
        """
        Get the grouped metadata dataframe.
        @param metadata_df: The metadata dataframe.
        @param exclude_templates: List of templates names to exclude.
        @return:
        """
        df = pd.read_csv(self.performance_summary_path)
        df = df[~df[ResultConstants.TEMPLATE_NAME].isin(exclude_templates)] if exclude_templates else df
        # add the ResultConstants.ACCURACY_COLUMN columns from df to the metadata_df by the template_name
        metadata_df = metadata_df.join(df.set_index(ResultConstants.TEMPLATE_NAME)[ResultConstants.ACCURACY_COLUMN],
                                       on=ResultConstants.TEMPLATE_NAME)
        if self.selected_best_value_axes:
            # group by the selected axes and calculate the mean of the accuracy
            grouped_metadata_df = metadata_df.groupby(self.selected_best_value_axes)[
                ResultConstants.ACCURACY_COLUMN].mean().rename(
                ResultConstants.ACCURACY_COLUMN)
            index_lists = (metadata_df.groupby(self.selected_best_value_axes).apply(lambda x: list(x.index)).rename(
                ResultConstants.TEMPLATE_NAME))
            grouped_metadata_with_index = grouped_metadata_df.to_frame().join(index_lists.to_frame()).reset_index()
        else:
            # if no axes are selected, take the mean of the accuracy
            grouped_metadata = {ResultConstants.ACCURACY_COLUMN: [metadata_df[ResultConstants.ACCURACY_COLUMN].mean()]}
            index_lists = metadata_df.index.tolist()
            grouped_metadata.update({ResultConstants.TEMPLATE_NAME: [index_lists]})
            grouped_metadata_with_index = pd.DataFrame(grouped_metadata)
        # remove rows that accuracy is NaN
        grouped_metadata_with_index = grouped_metadata_with_index.dropna(subset=[ResultConstants.ACCURACY_COLUMN])
        return grouped_metadata_with_index

    def get_best_row(self, grouped_metadata_df):
        best_row = grouped_metadata_df.loc[grouped_metadata_df[ResultConstants.ACCURACY_COLUMN].idxmax()]
        return best_row

    def choose_across_axes(self, metadata_df: pd.DataFrame) -> pd.Series:
        """
        Choose the best value for each axis separately -
        for each axis, choose the value with the highest accuracy *across* all the others values of the others axes.
        @param metadata_df: the metadata dataframe
        @return: the best row with the best values for each axis (the accuracy is NaN)
        """
        axes = list(metadata_df.columns.tolist() & ConfigParams.override_options.keys())
        best_values = {}
        for axis in axes:
            avg_values = metadata_df.groupby(axis)[ResultConstants.ACCURACY_COLUMN].mean()
            # choose the value with the highest accuracy
            best_value = avg_values.idxmax()
            best_values[axis] = best_value
        best_row = pd.Series(best_values)
        # add accuracy column with NaN value to best_row
        best_row[ResultConstants.ACCURACY_COLUMN] = None
        return best_row
