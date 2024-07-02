import argparse
import json
from pathlib import Path
from typing import List, Tuple

from termcolor import colored
from tqdm import tqdm

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.CreateData.LLMDataset import LLMDataset
from src.ModelsPredictors.LLMProcessor import LLMProcessor
from src.utils.Constants import Constants
from src.utils.ReadLLMParams import ReadLLMParams
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants


class LLMPredictor:
    def __init__(self, llmp: LLMProcessor,
                 predict_prob_of_tokens: bool = LLMProcessorConstants.PREDICT_PROB_OF_TOKENS,
                 batch_size: int = ExperimentConstants.BATCH_SIZE):
        """
        Initializes the LLMPredictor.
        @param llmp:
        @param batch_size:
        """
        self.llmp = llmp
        self.predict_prob_of_tokens = predict_prob_of_tokens
        self.batch_size = batch_size

    def predict_on_single_dataset(self, eval_dataset, eval_value: str, results_file_path: Path,
                                  possible_gt_tokens: List[str] = None) -> None:
        """
        Predict the model on a single dataset.

        @eval_set: The evaluation set name.
        @results_file_path: The name of the file to save the results in.

        """
        # find the max_new_tokens parameter from the eval_set (the maximum number of tokens in the target)
        max_new_tokens = max([len(instance["target"].split()) for instance in eval_dataset])
        max_new_tokens = max(max_new_tokens, 12)
        max_new_tokens = min(max_new_tokens, 25)

        eval_set_indexes = list(range(len(eval_dataset)))
        filter_eval_set, filter_eval_set_indexes = self.filter_saved_instances(eval_dataset, eval_value,
                                                                               eval_set_indexes,
                                                                               results_file_path)
        # print in red the number of instances that were already predicted and will be skipped
        print(colored(f"{len(eval_dataset)} instances were already predicted and will be skipped.", "red"))
        # print in green the number of instances that will be predicted
        print(colored(f"{len(filter_eval_set)} instances will be predicted.", "green"))
        loaded_data = self.load_results_file(results_file_path)
        # each result is a dictionary with the keys: 'idx', 'input_text', 'result', 'ground_truth'.
        # create a list of the indexes of the instances that were already predicted
        if eval_value in loaded_data['results']:
            loaded_results = loaded_data['results'][eval_value]
            loaded_idxs = [result['Index'] for result in loaded_results]
            loaded_input_texts = [result['Instance'] for result in loaded_results]
            loaded_ground_truths = [result['GroundTruth'] for result in loaded_results]
            loaded_answers = [result['Result'] for result in loaded_results]
            loaded_max_tokens_probability = [" " if 'MaxTokenProbability' not in result else result['MaxTokenProbability'] for result in loaded_results]

        else:
            loaded_idxs = []
            loaded_input_texts = []
            loaded_ground_truths = []
            loaded_answers = []
            loaded_max_tokens_probability = []

        # run the model on the dataset and save the results in the file after each batch
        counter_idx = 0
        for batch_start in tqdm(range(0, len(filter_eval_set), self.batch_size)):
            batch_instances = filter_eval_set[batch_start:batch_start + self.batch_size]
            batch_indexes = filter_eval_set_indexes[batch_start:batch_start + self.batch_size]

            # Iterate over instances in the batch

            counter_idx += self.batch_size
            input_text = [batch_instance["source"] for batch_instance in batch_instances]
            ground_truth = [batch_instance["target"] for batch_instance in batch_instances]
            generated_tokens_decoded, max_tokens = self.llmp.predict(input_text, self.predict_prob_of_tokens,
                                                                     max_new_tokens,
                                                                     possible_gt_tokens=possible_gt_tokens)
            loaded_idxs.extend(batch_indexes)
            loaded_input_texts.extend(input_text)
            loaded_ground_truths.extend(ground_truth)
            loaded_answers.extend(generated_tokens_decoded)
            loaded_max_tokens_probability.extend(max_tokens)
            self.save_results(results_file_path, eval_value, loaded_idxs, loaded_input_texts, loaded_answers,
                              loaded_ground_truths, loaded_max_tokens_probability, loaded_data)

    def predict_dataset(self, llm_dataset: LLMDataset, evaluate_on: list,
                        results_file_path: Path, possible_gt_tokens: List[str] = None) -> None:
        """
        Predict the model on all the instances in the dataset.

        """
        for eval_value in evaluate_on:
            if eval_value not in llm_dataset.dataset:
                raise ValueError(f"The evaluation set {eval_value} is not in the dataset.")

            else:
                eval_dataset = llm_dataset.dataset[eval_value]
                self.predict_on_single_dataset(eval_dataset, eval_value, results_file_path=results_file_path,
                                               possible_gt_tokens=possible_gt_tokens)

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
                     idxs: List[int], input_texts: List[str], results: List[str], ground_truths: List[str],
                     loaded_max_tokens_probability: List[str], loaded_data: dict) -> None:
        """
        Save the results in a JSON file.

        @param results_file_path: The name of the file to save the results in.
        @param eval_value: The evaluation set name.
        @param idxs: The indices of the instances.
        @param input_texts: The input texts of the instances.
        @param results: The model predictions.
        @param ground_truths: The ground truth of the instances.

        """

        entries = self.create_entries(idxs, input_texts, results, loaded_max_tokens_probability, ground_truths)
        loaded_data['results'][eval_value] = entries
        with open(results_file_path, "w") as f:
            json.dump(loaded_data, f)

    def create_entries(self, idxs: List[int], input_texts: List[str], results: List[str],
                       loaded_max_tokens_probability: List[str], ground_truths: List[str]):
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
            "MaxTokenProbability": max_token,
            "GroundTruth": ground_truth
        } for idx, instance, result, max_token, ground_truth in
            zip(idxs, input_texts, results, loaded_max_tokens_probability, ground_truths)]
        # sort the entries by the index
        entries = sorted(entries, key=lambda x: x["Index"])
        return entries

    def filter_saved_instances(self, eval_dataset: list, eval_value: str, eval_dataset_indexes: list,
                               results_file_path: Path) -> Tuple[list, list]:
        """
        Filter the instances that have already been saved in the results file.
        @param eval_dataset: The evaluation set.
        @param eval_value: The evaluation set name.
        @param eval_dataset_indexes: The indices of the instances in the evaluation set.
        @param results_file_path: The name of the file to save the results in.
        @return:
        """

        data = self.load_results_file(results_file_path)
        if eval_value not in data['results']:
            indexes = []
        else:
            results = data['results'][eval_value]
            indexes = [entry["Index"] for entry in results]

        # filter the eval set and the indexes
        filter_eval_set = []
        filter_eval_set_indexes = []
        for idx, instance in zip(eval_dataset_indexes, eval_dataset):
            if idx not in indexes:
                filter_eval_set.append(instance)
                filter_eval_set_indexes.append(idx)
        return filter_eval_set, filter_eval_set_indexes


# Execute the main function
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = ReadLLMParams.read_llm_params(args)

    args.add_argument("--card", type=str)
    args.add_argument("--system_format", type=str, default="unitxt")
    args.add_argument("--max_instances", type=int, default=ExperimentConstants.MAX_INSTANCES)
    args.add_argument("--template_num", type=int, default=ExperimentConstants.TEMPLATE_NUM)
    args.add_argument("--demos_taken_from", type=str, default=ExperimentConstants.DEMOS_TAKEN_FROM)
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
                                       demos_taken_from=args.demos_taken_from,
                                       num_demos=args.num_demos,
                                       demos_pool_size=args.demos_pool_size,
                                       max_instances=args.max_instances,
                                       template_name=args.template_name)

    llm_dataset = llm_dataset_loader.load()
    llm_proc = LLMProcessor(args.model_name)

    llm_pred = LLMPredictor(llm_proc)
    llm_pred.predict_dataset(llm_dataset, evaluate_on=["train", "test"],
                             results_file_path=Path("results.json"))
