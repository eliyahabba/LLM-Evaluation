from pathlib import Path
from typing import Callable

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
DatasetsConstants = Constants.DatasetsConstants


class ModelDatasetRunner:
    def __init__(self, structured_input_folder_path: str, evaluate_on: list):
        self.structured_input_folder_path = structured_input_folder_path
        self.evaluate_on = evaluate_on

    def run_function_on_all_models_and_datasets(self, processing_function: Callable,
                                                kwargs: dict = None
                                                ) -> None:
        results_folder = Path(self.structured_input_folder_path)
        eval_on = self.evaluate_on
        models_names = sorted([file for file in results_folder.glob("*") if file.is_dir()],
                              key=lambda x: x.name.lower())
        for model_name in models_names:
            datasets = sorted([file for file in model_name.glob("*") if file.is_dir()])
            for dataset_folder in datasets:
                shots = [file for file in dataset_folder.glob("*") if file.is_dir()]
                for shot in shots:
                    formats = [file for file in shot.glob("*") if file.is_dir()]
                    for format_folder in formats:
                        for eval_value in eval_on:
                            try:
                                processing_function(format_folder, eval_value, kwargs)
                            except Exception as e:
                                print(f"Error in {model_name.name}/{dataset_folder.name}/{shot.name}/"
                                      f"{format_folder.name} for {eval_value}: {e}")


# Example usage:
def check_comparison_matrix(folder, eval_value):
    print(f"Checking comparison matrix in {folder} for {eval_value}")


def check_results_files(folder, eval_value):
    print(f"Checking results files in {folder} for {eval_value}")


if __name__ == "__main__":
    results_folder = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH
    eval_on = ExperimentConstants.EVALUATE_ON
    runner = ModelDatasetRunner(results_folder, eval_on)
    runner.run_function_on_all_models_and_datasets(check_comparison_matrix)
    runner.run_function_on_all_models_and_datasets(check_results_files)
