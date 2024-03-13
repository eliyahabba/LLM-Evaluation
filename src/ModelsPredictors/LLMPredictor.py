import argparse
import json
from pathlib import Path

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.CreateData.LLMDataset import LLMDataset
from src.ModelsPredictors.LLMProcessor import LLMProcessor
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class LLMPredictor:
    def __init__(self, llmp: LLMProcessor):
        self.llmp = llmp

    def predict_on_single_dataset(self, eval_set, eval_value: str, results_file_path: Path):
        """
        Predict the model on a single dataset.

        @eval_set: The evaluation set name.
        @results_file_path: The name of the file to save the results in.

        @return: The list of prediction results for the dataset.
        """
        results = []
        for idx, instance in enumerate(eval_set):
            input_text = instance["source"]
            ground_truth = instance["target"]
            result = self.llmp.predict(input_text)
            self.save_results(results_file_path, eval_value, idx, input_text, result, ground_truth)
            results.append(result)
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

    def save_results(self, results_file_path: Path, eval_value: str, idx: int, instance: dict, result: str,
                     ground_truth: str) -> None:
        """
        Save the results in a JSON file.

        @param results_file_path: The name of the file to save the results in.
        @param idx: The index of the instance.
        @param instance: The instance dictionary.
        @param result: The prediction result string.
        @param ground_truth: The ground truth answer.

        """
        entry = {
            "Index": idx,
            "Instance": instance,
            "Result": result,
            "GroundTruth": ground_truth
        }

        with open(results_file_path, "r") as f:
            data = json.load(f)
            results = data['results']
            if eval_value in results:
                results[eval_value].append(entry)
            else:
                results[eval_value] = [entry]
        data['results'] = results
        with open(results_file_path, "w") as f:
            json.dump(data, f)


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
