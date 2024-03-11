import argparse
import json
from pathlib import Path

import numpy as np

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.CreateData.LLMDataset import LLMDataset
from src.ModelsPredictors.LLMPredictor import LLMPredictor
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class LLMEvaluator:
    def __init__(self, llmp: LLMPredictor):
        self.llmp = llmp

    def predict_on_single_dataset(self, eval_set, eval_value: str, file_name: Path):
        """
        Predict the model on a single dataset.

        @eval_set: The evaluation set name.
        @file_name: The name of the file to save the results in.

        @return: The list of prediction results for the dataset.
        """
        results = []
        for idx, instance in enumerate(eval_set):
            input_text = instance["source"]
            result = self.llmp.predict(input_text)
            self.save_results(file_name, eval_value, idx, input_text, result)
            results.append(result)
        return results

    def predict_dataset(self, llm_dataset: LLMDataset, evaluate_on: list,
                        results_file_name: Path) -> list:
        """
        Predict the model on all the instances in the dataset.

        """
        results = []
        for eval_value in evaluate_on:
            if eval_value not in llm_dataset.dataset:
                raise ValueError(f"The evaluation set {eval_value} is not in the dataset.")

            else:
                eval_dataset = llm_dataset.dataset[eval_value]
                result = self.predict_on_single_dataset(eval_dataset, eval_value,
                                                        file_name=results_file_name)
                results.append(result)
        return results

    def save_results(self, file_name: Path, eval_value: str, idx: int, instance: dict, result: str) -> None:
        """
        Save the results in a JSON file.

        @param idx: The index of the instance.
        @param instance: The instance dictionary.
        @param result: The prediction result string.
        """
        entry = {
            "Index": idx,
            "Instance": instance,
            "Result": result
        }

        with open(file_name, "r") as f:
            data = json.load(f)
            results = data['results']
            if eval_value in results:
                results[eval_value].append(entry)
            else:
                results[eval_value] = [entry]
        data['results'] = results
        with open(file_name, "w") as f:
            json.dump(data, f)

    def evaluate(self, dataset):
        """
        Evaluate the model on all the instances in the dataset with the given prompt.

        @param dataset: The dataset to evaluate the model on.
        """
        results = self.llmp.predict(dataset)
        # calculate the accuracy of the model with the dataset GT and the results
        answers = []
        for i in range(len(results)):
            answer = self.check_answer(results[i], dataset[i])
            answers.append(answer)

        # calculate the accuracy
        accuracy = np.mean(answers)
        return accuracy

    def check_answer(self, result, instance):
        """
        Check if the prediction matches the ground truth answer on multiple choice questions.

        @param result: The prediction result.
        @param instance: The instance dictionary.
        """
        # check if the result is in the choices
        if result in instance['target']:
            return 1
        else:
            return 0


# Execute the main function
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    args.add_argument("--card", type=str, default="cards.hellaswag")
    args.add_argument("--system_format", type=str, default="unitxt")
    args.add_argument("--max_instances", type=int, default=5)
    args.add_argument("--template_name", type=str, default="template_0")

    args = args.parse_args()

    # Save templates to local catalog
    catalog_manager = CatalogManager(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH)
    template = catalog_manager.load_from_catalog(args.template_name)

    llm_dataset_loader = DatasetLoader(card=args.card,
                                       template=template,
                                       system_format=args.system_format, max_instances=args.max_instances,
                                       template_name=args.template_name)

    llm_dataset = llm_dataset_loader.load()
    llmp = LLMPredictor(args.model_name)

    llm_eval = LLMEvaluator(llmp)
    results = llm_eval.predict_dataset(llm_dataset, args.evaluate_on)
