import json
from pathlib import Path

import evaluate
import pandas as pd
from tqdm import tqdm

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
metric = evaluate.load("unitxt/metric")


class EvaluateModel:
    def __init__(self, results_file: Path, eval_on: str):
        self.experiment = None
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

    def load_results_and_dataset(self):
        """
        Load the results from the json file.
        @return: list of results
        """
        self.load_experiment_file()

        template_name = f"{self.experiment['template_name']}"
        catalog_manager = CatalogManager(Utils.get_card_path(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH,
                                                             self.experiment['card']))
        template = catalog_manager.load_from_catalog(template_name)

        llm_dataset_loader = DatasetLoader(card=self.experiment['card'], template=template,
                                           system_format=self.experiment['system_format'],
                                           num_demos=self.experiment['num_demos'],
                                           demos_pool_size=self.experiment['demos_pool_size'],
                                           max_instances=self.experiment['max_instances'],
                                           template_name=self.experiment['template_name'])
        llm_dataset = llm_dataset_loader.load()
        return self.experiment['results'], llm_dataset

    def evaluate(self):
        """
        Calculate the scores of the model on the dataset.
        @return: the scores
        """
        results, llm_dataset = self.load_results_and_dataset()
        predictions, references = self.get_predictions_and_references(results, llm_dataset)
        scores = metric.compute(predictions=predictions, references=references)
        self.save_scores(scores)
        return scores

    def get_predictions_and_references(self, results, llm_dataset):
        results_to_eval = results[self.eval_on]
        predictions = [result['Result'] for result in results_to_eval]
        predictions_idx = [result['Index'] for result in results_to_eval]
        predictions = [" ".join(prediction) if isinstance(prediction, list) else prediction for prediction in
                       predictions]
        reference_dataset = llm_dataset.dataset[self.eval_on]
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
        file_path = folder_path / f"{card_name}_scores.csv"
        if not file_path.exists():
            scores_df.to_csv(file_path)
        else:
            # if the row already exists, dont add it again, just update the values
            if scores_df.index[0] in scores_df.index:
                current_scores_df = pd.read_csv(file_path, index_col=0)
                current_scores_df.loc[scores_df.index[0]] = scores_df.loc[scores_df.index[0]]
                current_scores_df.sort_index(inplace=True, key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
                current_scores_df.to_csv(file_path)
            else:
                scores_df.sort_index(inplace=True, key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
                scores_df.to_csv(file_path, mode='a', header=False)


if __name__ == "__main__":
    # Load the model and the dataset
    results_folder = ExperimentConstants.RESULTS_PATH
    eval_on = ExperimentConstants.EVALUATE_ON

    for dataset_folder in [file for file in results_folder.glob("*") if file.is_dir()]:
        for results_file in tqdm(dataset_folder.glob("*.json")):
            for eval_on in ExperimentConstants.EVALUATE_ON:
                eval_model = EvaluateModel(results_file, eval_on)
                results = eval_model.evaluate()
