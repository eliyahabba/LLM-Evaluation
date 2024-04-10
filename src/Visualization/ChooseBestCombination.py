from pathlib import Path
from typing import Tuple

import pandas as pd

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class ChooseBestCombination:
    def __init__(self, dataset_file_name: str, performance_summary_path: Path, selected_best_value_axes: list):
        self.dataset_file_name = dataset_file_name
        self.performance_summary_path = performance_summary_path
        self.selected_best_value_axes = selected_best_value_axes

    def choose_best_combination(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Choose the best combination of the values of the axes.
        @return: The best combination of the values of the axes.
        """
        templates_path = TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH
        metadata_file = templates_path / self.dataset_file_name / "templates_metadata.csv"
        metadata_df = pd.read_csv(metadata_file, index_col='template_name')
        grouped_metadata_df = self.get_grouped_metadata_df(metadata_df=metadata_df)
        best_row = self.get_best_row(grouped_metadata_df)
        return grouped_metadata_df, best_row

    def get_grouped_metadata_df(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        df = pd.read_csv(self.performance_summary_path)
        # add the 'accuracy' columns from df to the metadata_df by the template_name
        metadata_df = metadata_df.join(df.set_index('template_name')['accuracy'], on='template_name')
        if self.selected_best_value_axes:
            # group by the selected axes and calculate the mean of the accuracy
            grouped_metadata_df = metadata_df.groupby(self.selected_best_value_axes)[
                'accuracy'].mean().rename(
                'accuracy')
            index_lists = (metadata_df.groupby(self.selected_best_value_axes).apply(lambda x: list(x.index)).rename(
                'template_name'))
            grouped_metadata_with_index = grouped_metadata_df.to_frame().join(index_lists.to_frame()).reset_index()
        else:
            # if no axes are selected, take the mean of the accuracy
            grouped_metadata = {'accuracy': [metadata_df['accuracy'].mean()]}
            index_lists = metadata_df.index.tolist()
            grouped_metadata.update({'template_name': [index_lists]})
            grouped_metadata_with_index = pd.DataFrame(grouped_metadata)
        return grouped_metadata_with_index

    def get_best_row(self, grouped_metadata_df):
        best_row = grouped_metadata_df.loc[grouped_metadata_df['accuracy'].idxmax()]
        return best_row
