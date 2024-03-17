import json
from pathlib import Path
from typing import Tuple, Union

import evaluate
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
            json_data = json.load(f)
        self.experiment = json_data

    # def load_results(self) -> dict:
    #     """
    #     Load the results from the json file.
    #     @return: list of results
    #     """
    #     return self.experiment['results']
    #
    # def load_dataset(self) -> LLMDataset:
    #     """
    #     Load the dataset from the experiment.
    #
    #     @return: the dataset
    #     """
    #     template_name = f"{self.experiment['template_name']}"
    #
    #     catalog_manager = CatalogManager(Utils.get_card_path(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH,
    #                                                          self.experiment['card']))
    #     template = catalog_manager.load_from_catalog(template_name)
    #
    #     llm_dataset_loader = DatasetLoader(card=self.experiment['card'], template=template,
    #                                        system_format=self.experiment['system_format'],
    #                                        num_demos=self.experiment['num_demos'],
    #                                        demos_pool_size=self.experiment['demos_pool_size'],
    #                                        max_instances=self.experiment['max_instances'],
    #                                        template_name=self.experiment['template_name'])
    #     llm_dataset = llm_dataset_loader.load()
    #     return llm_dataset

    def load_results(self) -> dict:
        """
        Load the results from the json file.
        @return: list of results
        """
        self.load_experiment_file()
        results = self.experiment['results']
        # llm_dataset = self.load_dataset()
        return results

    def evaluate(self, results: dict, llm_dataset: LLMDataset) -> Union[None, dict]:
        """
        Calculate the scores of the model on the dataset.
        @return: the scores
        """
        if self.eval_on_value not in results:
            return None
        predictions, references = self.get_predictions_and_references(results, llm_dataset)
        scores = metric.compute(predictions=predictions, references=references)
        self.save_scores(scores)
        return scores

    def get_predictions_and_references(self, results: dict, llm_dataset: LLMDataset):
        results_to_eval = results[self.eval_on_value]
        predictions = [result['Result'] for result in results_to_eval]
        predictions_idx = [result['Index'] for result in results_to_eval]
        predictions = [" ".join(prediction) if isinstance(prediction, list) else prediction for prediction in
                       predictions]
        reference_dataset = llm_dataset.dataset[self.eval_on_value]
        # get the references for the predictions that were made
        reference_dataset = [reference_dataset[idx] for idx in predictions_idx]
        return predictions, reference_dataset

    def save_scores(self, scores: dict):
        """
        Save the scores to a csv file with the metadata of the experiment.
        """
        metadata_columns = ['card', 'template_name', 'system_format', 'num_demos', 'demos_pool_size', 'max_instances']
        scores_columns = list(scores[0]['score']['global'].keys())
        columns = metadata_columns + scores_columns + ['number_of_instances']
        metadata_values = [self.experiment[metadata] for metadata in metadata_columns]
        scores_values = [scores[0]['score']['global'][score_name] for score_name in scores_columns]
        scores_values = [f"{score:.3f}" if isinstance(score, float) else score for score in scores_values]
        scores_df = pd.DataFrame([metadata_values + scores_values + [len(scores)]], columns=columns)
        # take the template column to be the row index
        scores_df.set_index('template_name', inplace=True)

        card_name = scores_df['card'].values[0]
        folder_path = self.results_file.parent
        file_path = folder_path / f"{card_name}_scores_{self.eval_on_value}_data.csv"
        if not file_path.exists():
            scores_df.to_csv(file_path)
        else:
            # if the row already exists, dont add it again, just update the values
            if scores_df.index[0] in scores_df.index:
                # read all the columns as object to avoid the error of mixing types
                current_scores_df = pd.read_csv(file_path, index_col=0, dtype=object)
                current_scores_df.loc[scores_df.index[0]] = scores_df.loc[scores_df.index[0]]
                current_scores_df.sort_index(inplace=True,
                                             key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
                current_scores_df.to_csv(file_path)
            else:
                scores_df.sort_index(inplace=True, key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
                scores_df.to_csv(file_path, mode='a', header=False)


def read_experiment(results_file: Path) -> dict:
    """
    Load the dataset from the experiment.

    @return: the dataset
    """
    with open(results_file, "r") as f:
        experiment = json.load(f)
    return experiment


def get_experiment_values(experiment: dict) -> dict:
    """
    """

    metadata_experiment = ['card', 'system_format', 'num_demos', 'demos_pool_size', 'max_instances']
    return {metadata: experiment[metadata] for metadata in metadata_experiment}


def compare_experiments_files(experiment_file1: Path, experiment_file2: Path) -> bool:
    """
    """
    experiment1 = read_experiment(experiment_file1)
    experiment2 = read_experiment(experiment_file2)

    experiment1_values = get_experiment_values(experiment1)
    experiment2_values = get_experiment_values(experiment2)
    return experiment1_values == experiment2_values


def load_dataset(results_file: Path) -> LLMDataset:
    """
    Load the dataset from the experiment.
    """
    experiment = read_experiment(results_file)
    template_name = f"{experiment['template_name']}"
    catalog_manager = CatalogManager(Utils.get_card_path(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH,
                                                         experiment['card']))
    template = catalog_manager.load_from_catalog(template_name)

    llm_dataset_loader = DatasetLoader(card=experiment['card'], template=template,
                                       system_format=experiment['system_format'],
                                       num_demos=experiment['num_demos'],
                                       demos_pool_size=experiment['demos_pool_size'],
                                       max_instances=experiment['max_instances'],
                                       template_name=experiment['template_name'])
    llm_dataset = llm_dataset_loader.load()
    return llm_dataset


if __name__ == "__main__":
    # Load the model and the dataset
    results_folder = ExperimentConstants.RESULTS_PATH
    eval_on = ExperimentConstants.EVALUATE_ON

    for dataset_folder in [file for file in results_folder.glob("*") if file.is_dir()]:
        for shot in [file for file in dataset_folder.glob("*") if file.is_dir()]:
            # use the first file in the folder of the current dataset to compare the experiments
            first_result_file = next(shot.glob("*.json"))
            llm_dataset = load_dataset(first_result_file)
            for results_file in tqdm(shot.glob("*.json")):
                # check that the experiment is the same as the first one
                experiment = read_experiment(results_file)
                if not compare_experiments_files(results_file, first_result_file):
                    print(f"The experiment in {results_file} is not the same as the first one.")
                    continue
                for eval_on_value in ExperimentConstants.EVALUATE_ON:
                    try:
                        eval_model = EvaluateModel(results_file, eval_on_value)
                        results = eval_model.load_results()
                        results = eval_model.evaluate(results, llm_dataset)
                    except Exception as e:
                        print(f"Error in {results_file}: {e}")
                        continue
