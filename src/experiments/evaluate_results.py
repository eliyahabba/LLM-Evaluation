import os
os.environ["UNITXT_ALLOW_UNVERIFIED_CODE"] = "True"
import argparse
import json
from pathlib import Path
from typing import Union

import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.experiments.data_loading.CatalogManager import CatalogManager
from src.experiments.data_loading.DatasetLoader import DatasetLoader
from src.experiments.data_loading.NLPDataset import NLPDataset
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
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

    def save_results_to_experiment_file(self, results: dict):
        """
        Save the results to the experiment file.
        """
        self.experiment['results'] = results
        with open(self.results_file, "w") as f:
            json.dump(self.experiment, f)

    def evaluate(self, results: dict, llm_dataset: NLPDataset, save_to_file: bool = True) -> Union[None, dict]:
        """
        Calculate the scores of the model on the dataset.
        @return: the scores
        """
        if self.eval_on_value not in results:
            return None
        predictions, references, predictions_idx = self.get_predictions_and_references(results, llm_dataset)
        scores = metric.compute(predictions=predictions, references=references)
        # add the scores to the results
        for idx, result in enumerate(results[self.eval_on_value]):
            result['Score'] = scores[idx]['score']['instance']['accuracy']
        self.save_results_to_experiment_file(results)
        if save_to_file:
            self.save_scores(scores)
        scores_by_index = self.parser_predictions(scores, predictions_idx, len(llm_dataset.dataset[self.eval_on_value]))
        return scores_by_index

    def get_predictions_and_references(self, results: dict, llm_dataset: NLPDataset):
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
        folder_path = self.results_file.parent
        file_path = folder_path / f"performance_summary_{self.eval_on_value}_data.csv"
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


def load_dataset(results_file: Path, catalog_path: Path, loaded_datasets: dict) -> NLPDataset:
    """
    Load the dataset from the experiment.
    """
    experiment = read_experiment(results_file)
    template_name = f"{experiment['template_name']}"
    catalog_manager = CatalogManager(catalog_path)
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
    if not experiment['results']:
        return None
    llm_dataset_loader = DatasetLoader(card=experiment['card'], template=template,
                                       system_format=experiment['system_format'],
                                       demos_taken_from=experiment.get('demos_taken_from', 'validation'),
                                       num_demos=experiment['num_demos'],
                                       demos_pool_size=experiment['demos_pool_size'] if "mmlu" not in experiment[
                                           "card"] or experiment['num_demos'] != 0 else None,
                                       max_instances=max(
                                           max([index['Index'] for index in experiment['results'][key]]) for key in
                                           experiment['results']) + 1,
                                       template_name=experiment['template_name'])
    llm_dataset = llm_dataset_loader.load()
    loaded_datasets[template_hash] = llm_dataset
    return llm_dataset


if __name__ == "__main__":
    # Load the model and the dataset
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="LLAMA13B")
    args.add_argument("--results_folder", type=str,
                      default=TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_TOPIC_FOLDER_NAME)
    args.add_argument("--eval_on", type=str, default=ExperimentConstants.EVALUATE_ON_ANALYZE)
    args = args.parse_args()
    model_name = LLMProcessorConstants.BASE_MODEL_NAMES[args.model_name].split('/')[1]
    model_path = ExperimentConstants.MAIN_RESULTS_PATH / args.results_folder / model_name
    catalog_path = TemplatesGeneratorConstants.CATALOG_PATH
    args.results_folder = ExperimentConstants.MAIN_RESULTS_PATH / Path(args.results_folder)

    # models_names = [models_name for models_name in models_names if "Lla" in str(models_name)]
    error_files, errors_msgs = [], []
    dataset_sizes = pd.read_csv(TemplatesGeneratorConstants.DATASET_SIZES_PATH)
    print("Models to evaluate: ", model_path)
    datasets = sorted([file for file in model_path.glob("*") if file.is_dir()])
    # datasets = [dataset for dataset in datasets if "mmlu" in str(dataset)]
    # datasets = datasets[::-1]
    datasets = [dataset for dataset in datasets if "hell" in str(dataset)]
    for dataset_folder in datasets:
        print(f"Start evaluating {dataset_folder.name}")
        car_dataset_sizes = dataset_sizes[
            dataset_sizes["Name"] == dataset_folder.name]
        shots = [file for file in dataset_folder.glob("*") if file.is_dir()]
        # shots = [shot for shot in shots if "one" in str(shot)]
        loaded_datasets = {}
        for shot in shots:
            formats = [file for file in shot.glob("*") if file.is_dir()]
            for format_folder in formats:
                results_files = sorted([file for file in format_folder.glob("*.json")],
                                       key=lambda x: int(x.name.split(".json")[0].split("_")[-1]))
                # results_files = [file for file in results_files if "template_2" in str(file)]
                # results_files = results_files[:1]

                summary_of_accuracy_results = {eval_on_value: pd.DataFrame() for eval_on_value in args.eval_on}
                for eval_on_value in args.eval_on:
                    size = car_dataset_sizes.iloc[0][eval_on_value]

                    comparison_matrix_file = format_folder / f"comparison_matrix_{eval_on_value}_data.csv"
                    for results_file in tqdm(results_files):
                        try:
                            eval_model = EvaluateModel(results_file, eval_on_value)
                            results = eval_model.load_results_from_experiment_file()
                            results_file_number = results_file.name.split(".json")[0]
                            if comparison_matrix_file.exists():
                                try:
                                    comparison_df = pd.read_csv(comparison_matrix_file)
                                    if results_file_number in comparison_df.columns and \
                                            not comparison_df[results_file_number].isna().any() and \
                                            size == comparison_df[results_file_number].shape[0] and \
                                            all(['Score' in result for result in results[eval_on_value]]):
                                        continue
                                except pd.errors.EmptyDataError:
                                    # delete the file if it is empty
                                    comparison_matrix_file.unlink()
                            llm_dataset = load_dataset(results_file, catalog_path, loaded_datasets)
                            if llm_dataset is None:
                                continue
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
                    if results_df.empty:
                        continue
                    # sort the columns by the number of the template that in the columns name
                    results_df = results_df.reindex(sorted(results_df.columns, key=lambda x: int(x.split("_")[-1])),
                                                    axis=1)
                    comparison_matrix_file = format_folder / f"comparison_matrix_{eval_on_value}_data.csv"
                    if comparison_matrix_file.exists():
                        comparison_matrix = pd.read_csv(comparison_matrix_file)
                        # add from comparison matrix only the new columns that are not in the results_df
                        comparison_matrix = comparison_matrix[
                            [col for col in comparison_matrix.columns if col not in results_df.columns]]
                        results_df = pd.concat([results_df, comparison_matrix], axis=1)
                    # sort the columns by the number of the template that in the columns name
                    results_df = results_df.reindex(sorted(results_df.columns, key=lambda x: int(x.split("_")[-1])),
                                                    axis=1)
                    results_df.to_csv(comparison_matrix_file, index=False)
                    # print the size of the results vs the number of the templates
                    print(f"Results size: {results_df.shape[0]}")
                    print(f"Number of templates: {results_df.shape[1]}")
    for file, error in zip(error_files, errors_msgs):
        print(error)
        print(file)
