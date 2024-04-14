from abc import abstractmethod
from pathlib import Path

import pandas as pd

from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants

MAIN_RESULTS_PATH = ExperimentConstants.MAIN_RESULTS_PATH
RESULTS_FOLDER = "structured_input"
SHOT = "zero_shot"
FORMAT = "empty_system_format"


class Clustering:
    def __init__(self, model: str, dataset: str, eval_value: str,
                 main_results_folder: str = MAIN_RESULTS_PATH):
        self.labels = None
        self.data = None
        self.comparison_matrix_df = None

        self.main_results_folder = main_results_folder
        self.results_folder = f"{main_results_folder}/{RESULTS_FOLDER}/{model}/{dataset}/{SHOT}/{FORMAT}"
        self.eval_value = eval_value

    def create_results_file(self, index_name=str) -> pd.DataFrame:
        # Format results as a DataFrame
        columns = list(map(lambda x: x.split("experiment_")[1], self.comparison_matrix_df.columns.values))
        results = pd.DataFrame([self.labels],
                               columns=columns, index=[index_name]).T
        results.index.name = "template_name"
        results.reset_index(inplace=True, drop=False)
        return results

    def load_comparison_matrix(self) -> None:
        """
        Load the results from the specified path.
        @return:
        """
        file_name = f"{ResultConstants.COMPARISON_MATRIX}_{self.eval_value}_data.csv"
        file_path = f"{self.results_folder}/{file_name}"
        self.comparison_matrix_df = pd.read_csv(file_path)
        self.data = self.comparison_matrix_df.to_numpy()
        # transpose the data, so that the templates are the rows and the features are the columns
        self.data = self.data.T

    def get_result_output_path(self, cluster_method) -> Path:
        """
        Get the path to the output file.
        @return: The path to the output file.
        """
        file_path = Path(
            f"{self.results_folder}/{cluster_method}_{ResultConstants.CLUSTERING_RESULTS}_{self.eval_value}_data.csv")
        return file_path

    def save_results(self, results: pd.DataFrame, file_path:Path, column_name: str) -> None:
        """
        Save the results to the specified path.
        @param results: The results to be saved.
        @param file_path: The path to the output file.
        @param column_name: The name of the column in the results DataFrame.
        @return: None
        """
        # if the file already exists, read the file, and append the new results (if they don't already exist,
        # if they do, update them)
        if file_path.exists():
            existing_results = pd.read_csv(file_path, index_col=0)
            # check if there is column with the same name as the new results Cluster column
            if column_name in existing_results.columns:
                # update the results
                existing_results[column_name] = results[column_name]
                updated_results = existing_results
            else:
                # concat on all the columns
                updated_results = pd.merge(existing_results, results, on='template_name', how='inner')
            # sort the columns
            updated_results = updated_results.reindex(sorted(updated_results.columns), axis=1)
            # out the "template_name" column as the first column
            updated_results = updated_results[["template_name"] + [col for col in updated_results.columns if col != "template_name"]]
            updated_results.to_csv(file_path)
        else:
            results.to_csv(file_path)

    @abstractmethod
    def fit(self):
        pass
