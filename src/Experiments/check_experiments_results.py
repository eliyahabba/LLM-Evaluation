import json
from pathlib import Path

from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants



class ExperimentsResults:
    def __init__(self, results_file: Path, eval_on: str):
        self.results_file = results_file
        self.eval_on = eval_on

    def load_experiment_file(self):
        """
        Load the results from the json file.
        @return: list of results
        """
        with open(self.results_file, "r") as f:
            json_data = json.load(f)
        self.experiment = json_data

    def count_results(self):
        """
        Count the results of the model on the dataset.
        @return:
        """
        self.load_experiment_file()
        result_counter = {}
        results = self.experiment['results']
        for eval_on_value in self.eval_on:
            if eval_on_value not in results:
                result_counter[eval_on_value] = 0
                continue
            results_to_eval = results[eval_on_value]
            predictions_idx = [result['Index'] for result in results_to_eval]
            result_counter[eval_on_value] = len(predictions_idx)

        return result_counter


if __name__ == "__main__":
    # Load the model and the dataset
    results_folder = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH
    eval_on = ExperimentConstants.EVALUATE_ON
    models_names = [file for file in results_folder.glob("*") if file.is_dir()]
    for model_name in models_names:
        datasets = [file for file in model_name.glob("*") if file.is_dir()]
        for dataset_folder in datasets:
            shots = [file for file in dataset_folder.glob("*") if file.is_dir()]
            for shot in shots:
                formats = [file for file in shot.glob("*") if file.is_dir()]
                for format_folder in formats:
                    results_files = list([file for file in format_folder.glob("*.json")])
                    sorted_file_paths = sorted(results_files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
                    for results_file in sorted_file_paths:
                        try:
                            experiments_results = ExperimentsResults(results_file, eval_on)
                            results_counter = experiments_results.count_results()
                            # if one of the values of results_counter < 100 then print the results_counter
                            if any([value < 100 for value in results_counter.values()]):
                                print(f"{results_file}: {results_counter}")
                        except Exception as e:
                            print(f"Error in {results_file}: {e}")
                            continue
