import json
from pathlib import Path
from typing import Union

import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.CreateData.LLMDataset import LLMDataset
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
metric = evaluate.load("unitxt/metric")


class EvaluateModel:
    def __init__(self, results_file: Path, eval_on_value: str):
        self.experiment = None
        self.results_file = results_file
        self.eval_on_value = eval_on_value

    def load_experiment_file(self):
        """
        Load the results from the json file.
        @return: list of results
        """
        with open(self.results_file, "r") as f:
            self.experiment = json.load(f)

    def load_results_from_experiment_file(self) -> dict:
        """
        Load the results from the json file.
        @return: list of results
        """
        self.load_experiment_file()
        results = self.experiment['results']
        return results

    def evaluate(self, results: dict, llm_dataset: LLMDataset, save_to_file: bool = True) -> Union[None, dict]:
        """
        Calculate the scores of the model on the dataset.
        @return: the scores
        """
        if self.eval_on_value not in results:
            return None
        predictions, references, predictions_idx = self.get_predictions_and_references(results, llm_dataset)
        scores = metric.compute(predictions=predictions, references=references)
        if save_to_file:
            self.save_scores(scores)
        scores_by_index = self.parser_predictions(scores, predictions_idx, len(llm_dataset.dataset[self.eval_on_value]))
        return scores_by_index

    def get_predictions_and_references(self, results: dict, llm_dataset: LLMDataset):
        results_to_eval = results[self.eval_on_value]
        predictions = [result['Result'] for result in results_to_eval]
        predictions_idx = [result['Index'] for result in results_to_eval]
        predictions = [" ".join(prediction) if isinstance(prediction, list) else prediction for prediction in
                       predictions]
        reference_dataset = llm_dataset.dataset[self.eval_on_value]
        # get the references for the predictions that were made
        reference_dataset = [reference_dataset[idx] for idx in predictions_idx]
        return predictions, reference_dataset, predictions_idx

    def create_row_from_metadata(self) -> dict:
        """
        Get the metadata of the experiment.
        """
        metadata_columns = ['card', 'template_name', 'system_format', 'num_demos', 'demos_pool_size', 'max_instances']
        return {metadata: self.experiment[metadata] for metadata in metadata_columns}

    def create_row_from_scores(self, scores: dict) -> dict:
        """
        Get the columns of the scores.
        """
        scores_columns = list(scores[0]['score']['global'].keys())
        scores_values = [scores[0]['score']['global'][score_name] for score_name in scores_columns]
        scores_values = [f"{score:.3f}" if isinstance(score, float) else score for score in scores_values]
        # add to the scores the number of results
        scores_columns.append('number_of_instances')
        scores_values.append(len(scores))
        return {score: value for score, value in zip(scores_columns, scores_values)}

    def save_scores(self, scores: dict):
        """
        Save the scores to a csv file with the metadata of the experiment.
        """
        row_metadata = self.create_row_from_metadata()
        row_scores = self.create_row_from_scores(scores)

        scores_df = self._create_scores_dataframe(row_metadata, row_scores)
        self._write_scores_to_file(scores_df)

    def _create_scores_dataframe(self, metadata: dict, scores: dict) -> pd.DataFrame:
        """
        Create a DataFrame containing evaluation scores.
        """
        scores_df = pd.DataFrame([{**metadata, **scores}])
        scores_df.set_index('template_name', inplace=True)
        return scores_df

    def _write_scores_to_file(self, scores_df: pd.DataFrame) -> None:
        """
        Write the evaluation scores to a CSV file.
        """
        card_name = scores_df['card'].values[0]
        folder_path = self.results_file.parent
        file_path = folder_path / f"{card_name}_scores_{self.eval_on_value}_data.csv"
        if not file_path.exists():
            scores_df.to_csv(file_path)
        else:
            if scores_df.index[0] in scores_df.index:
                current_scores_df = pd.read_csv(file_path, index_col=0, dtype=object)
                current_scores_df.loc[scores_df.index[0]] = scores_df.loc[scores_df.index[0]]
                current_scores_df.sort_index(inplace=True,
                                             key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
                current_scores_df.to_csv(file_path)
            else:
                scores_df.sort_index(inplace=True, key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
                scores_df.to_csv(file_path, mode='a', header=False)

    def parser_predictions(self, scores, predictions_idx, num_of_total_instances):
        scores_by_index = np.nan * np.zeros(num_of_total_instances)
        for idx, score in zip(predictions_idx, scores):
            scores_by_index[idx] = 1 if score['score']['instance']['accuracy'] == 1 else 0
        return scores_by_index
        pass


def read_experiment(results_file: Path) -> dict:
    """
    Load the dataset from the experiment.

    @return: the dataset
    """
    with open(results_file, "r") as f:
        experiment = json.load(f)
    return experiment


def load_dataset(results_file: Path, loaded_datasets: dict) -> LLMDataset:
    """
    Load the dataset from the experiment.
    """
    experiment = read_experiment(results_file)
    template_name = f"{experiment['template_name']}"
    catalog_manager = CatalogManager(Utils.get_card_path(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH,
                                                         experiment['card']))
    template = catalog_manager.load_from_catalog(template_name)
    if experiment['num_demos'] == 0:
        template.postprocessors = [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
        template.target_choice_format = "{choice_numeral}. {choice_text}"

    template_hash = str(template.enumerator) + str(template.target_choice_format) + str(experiment['num_demos'])
    if template_hash in loaded_datasets:
        return loaded_datasets[template_hash]

    llm_dataset_loader = DatasetLoader(card=experiment['card'], template=template,
                                       system_format=experiment['system_format'],
                                       num_demos=experiment['num_demos'],
                                       demos_pool_size=experiment['demos_pool_size'],
                                       max_instances=experiment['max_instances'],
                                       template_name=experiment['template_name'])
    llm_dataset = llm_dataset_loader.load()
    loaded_datasets[template_hash] = llm_dataset
    return llm_dataset


if __name__ == "__main__":
    # Load the model and the dataset
    results_folder = ExperimentConstants.RESULTS_PATH
    eval_on = ExperimentConstants.EVALUATE_ON
    # eval_on = [ 'test', 'train']
    datasets = [file for file in results_folder.glob("*") if file.is_dir()]
    # datasets = [dataset for dataset in datasets if "race" in str(dataset)]
    error_files, errors_msgs = [], []

    for dataset_folder in datasets:
        shots = [file for file in dataset_folder.glob("*") if file.is_dir()]
        # shots = [shot for shot in shots if "one" in str(shot)]
        loaded_datasets = {}
        for shot in shots:
            results_files = [file for file in shot.glob("*.json")]
            # results_files = [file for file in results_files if "template_2" in str(file)]

            summary_of_accuracy_results = {eval_on_value: pd.DataFrame() for eval_on_value in eval_on}
            for results_file in tqdm(results_files):
                for eval_on_value in eval_on:
                    try:
                        llm_dataset = load_dataset(results_file, loaded_datasets)
                        eval_model = EvaluateModel(results_file, eval_on_value)
                        results = eval_model.load_results_from_experiment_file()
                        scores_by_index = eval_model.evaluate(results, llm_dataset)
                        if scores_by_index is not None:
                            scores_by_index_series = pd.Series(scores_by_index, name=results_file.stem)
                            # add the scores to the cumsum df so that the name of the file will be the index
                            summary_of_accuracy_results[eval_on_value] = pd.concat(
                                [summary_of_accuracy_results[eval_on_value], scores_by_index_series], axis=1)
                    except Exception as e:
                        error_files.append(results_file)
                        errors_msgs.append(e)
                        print(f"Error in {results_file}: {e}")
                        continue
            for eval_on_value, results_df in summary_of_accuracy_results.items():
                # sort the columns by the number of the template that in the columns name
                results_df = results_df.reindex(sorted(results_df.columns, key=lambda x: int(x.split("_")[-1])), axis=1)
                results_df.to_csv(shot / f"{eval_on_value}_accuracy_results.csv", index=False)
    for file, error in zip(error_files, errors_msgs):
        print(error)
        print(file)
