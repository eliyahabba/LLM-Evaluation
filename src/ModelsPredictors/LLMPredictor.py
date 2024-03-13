import argparse
import json
from pathlib import Path
from typing import List, Tuple

from termcolor import colored

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.CreateData.LLMDataset import LLMDataset
from src.ModelsPredictors.LLMProcessor import LLMProcessor
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class LLMPredictor:
    def __init__(self, llmp: LLMProcessor, batch_size: int = ExperimentConstants.BATCH_SIZE):
        """
        Initializes the LLMPredictor.
        @param llmp:
        @param batch_size:
        """
        self.llmp = llmp
        self.batch_size = batch_size

    def predict_on_single_dataset(self, eval_set, eval_value: str, results_file_path: Path):
        """
        Predict the model on a single dataset.

        @eval_set: The evaluation set name.
        @results_file_path: The name of the file to save the results in.

        @return: The list of prediction results for the dataset.
        """
        eval_set_indexes = list(range(len(eval_set)))
        filter_eval_set, filter_eval_set_indexes = self.filter_saved_instances(eval_set, eval_value, eval_set_indexes,
                                                                               results_file_path)
        # print in red the number of instances that were already predicted and will be skipped
        print(colored(f"{len(eval_set)} instances were already predicted and will be skipped.", "red"))
        # print in green the number of instances that will be predicted
        print(colored(f"{len(filter_eval_set)} instances will be predicted.", "green"))
        results = []
        # run the model on the dataset and save the results in the file after each batch
        idxs = []
        input_texts = []
        ground_truths = []

        for idx, instance in zip(filter_eval_set_indexes, filter_eval_set):
            input_text = instance["source"]
            ground_truth = instance["target"]
            result = self.llmp.predict(input_text)
            idxs.append(idx)
            input_texts.append(input_text)
            ground_truths.append(ground_truth)
            results.append(result)
            if idx % self.batch_size == 0:
                self.save_results(results_file_path, eval_value, idxs, input_texts, results, ground_truths)
                idxs = []
                input_texts = []
                ground_truths = []
                results = []

        return results

    def predict_dataset(self, llm_dataset: LLMDataset, evaluate_on: list,
                        results_file_path: Path) -> list:
        """
        Predict the model on all the instances in the dataset.

        """
        results = []
        for eval_value in evaluate_on:
            if eval_value not in llm_dataset.dataset:
                raise ValueError(f"The evaluation set {eval_value} is not in the dataset.")

            else:
                eval_dataset = llm_dataset.dataset[eval_value]
                result = self.predict_on_single_dataset(eval_dataset, eval_value, results_file_path=results_file_path)
                results.append(result)
        return results

    def load_results_file(self, results_file_path: Path) -> dict:
        """
        Load the results from a JSON file.
        @param results_file_path: The name of the file to load the results from.
        @return: The dictionary containing the results.
        """
        with open(results_file_path, "r") as f:
            data = json.load(f)
        return data

    def save_results(self, results_file_path: Path, eval_value: str,
                     idxs: List[int], input_texts: List[str], results: List[str], ground_truths: List[str]) -> None:
        """
        Save the results in a JSON file.

        @param results_file_path: The name of the file to save the results in.
        @param eval_value: The evaluation set name.
        @param idxs: The indices of the instances.
        @param input_texts: The input texts of the instances.
        @param results: The model predictions.
        @param ground_truths: The ground truth of the instances.



        """

        data = self.load_results_file(results_file_path)
        results = data['results']
        entries = self.create_entries(idxs, input_texts, results, ground_truths)
        if eval_value in results:
            results[eval_value].extend(entries)
        else:
            results[eval_value] = entries
        data['results'] = results
        with open(results_file_path, "w") as f:
            json.dump(data, f)

    def create_entries(self, idxs: List[int], input_texts: List[str], results: List[str], ground_truths: List[str]):
        """
        Create the entries for the results file.
        @param idxs: The indices of the instances.
        @param input_texts: The input texts of the instances.
        @param results: The model predictions.
        @param ground_truths: The ground truth of the instances.
        @return: The list of entries.
        """
        entries = [{
            "Index": idx,
            "Instance": instance,
            "Result": result,
            "GroundTruth": ground_truth
        } for idx, instance, result, ground_truth in zip(idxs, input_texts, results, ground_truths)]
        return entries

    def filter_saved_instances(self, eval_set: list, eval_value: str, eval_set_indexes: list,
                               results_file_path: Path) -> Tuple[list, list]:
        """
        Filter the instances that have already been saved in the results file.
        @param eval_set: The evaluation set.
        @param eval_value: The evaluation set name.
        @param eval_set_indexes: The indices of the instances in the evaluation set.
        @param results_file_path: The name of the file to save the results in.
        @return:
        """

        data = self.load_results_file(results_file_path)
        if eval_value not in data['results']:
            return eval_set, eval_set_indexes
        results = data['results'][eval_value]
        indexes = [entry["Index"] for entry in results]

        # filter the eval set and the indexes
        filter_eval_set = []
        filter_eval_set_indexes = []
        for idx, instance in zip(eval_set_indexes, eval_set):
            if idx not in indexes:
                filter_eval_set.append(instance)
                filter_eval_set_indexes.append(idx)
        return filter_eval_set, filter_eval_set_indexes


# Execute the main function
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    args.add_argument("--card", type=str)
    args.add_argument("--system_format", type=str, default="unitxt")
    args.add_argument("--max_instances", type=int, default=ExperimentConstants.MAX_INSTANCES)
    args.add_argument("--template_num", type=int, default=ExperimentConstants.TEMPLATE_NUM)
    args.add_argument("--num_demos", type=int, default=ExperimentConstants.NUM_DEMOS)
    args.add_argument("--demos_pool_size", type=int, default=ExperimentConstants.DEMOS_POOL_SIZE)

    args = args.parse_args()

    # Save templates to local catalog
    catalog_manager = CatalogManager(Utils.get_card_path(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH,
                                                         args.card))
    template = catalog_manager.load_from_catalog(args.template_name)

    llm_dataset_loader = DatasetLoader(card=args.card,
                                       template=template,
                                       system_format=args.system_format,
                                       num_demos=args.num_demos,
                                       demos_pool_size=args.demos_pool_size,
                                       max_instances=args.max_instances,
                                       template_name=args.template_name)

    llm_dataset = llm_dataset_loader.load()
    llm_proc = LLMProcessor(args.model_name)

    llm_pred = LLMPredictor(llm_proc)
    results = llm_pred.predict_dataset(llm_dataset, evaluate_on=["train", "test"],
                                       results_file_path=Path("results.json"))
