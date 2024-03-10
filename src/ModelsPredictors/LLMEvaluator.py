import argparse

import numpy as np

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.CreateData.LLMDataset import LLMDataset
from src.DataModels.LLMPredictor import LLMPredictor
from src.utils.Constants import Constants

UnitxtDataConstants = Constants.UnitxtDataConstants


class LLMEvaluator:
    def __init__(self, llmp: LLMPredictor, llm_dataset: LLMDataset):
        self.llmp = llmp
        self.llm_dataset = llm_dataset

    def predict_dataset(self) -> list:
        """
        Predict the model on all the instances in the dataset.

        """
        results = []
        for idx, instance in enumerate(self.llm_dataset.dataset):
            result = self.llmp.predict(instance)
            self.save_results(idx, instance, result)
            results.append(result)
        return results

    def save_results(self, idx: int, instance: dict, result: str) -> None:
        """
        Save the results in a json file.
        """
        data_name = self.llm_dataset.data_name
        with open(f"results_{data_name}.json", "a") as f:
            f.write(f"Index: {idx}\n")
            f.write(f"Instance: {instance}\n")
            f.write(f"Result: {result}\n")

    def evaluate(self, dataset):
        """
        Evaluate the model on all the instances in the dataset with the given prompt.
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
    args.add_argument("--card", type=str, default="cards.ai2_arc.arc_challenge")
    args.add_argument("--num_demos", type=int, default=5)
    args.add_argument("--demos_pool_size", type=int, default=5)
    args.add_argument("--system_format", type=str, default="unitxt")
    args.add_argument("--max_train_instances", type=int, default=5)

    args.add_argument("--template_name", type=str, default="template_0")
    args.add_argument("--data_path", type=str, default=UnitxtDataConstants.DATA_PATH,
                      help="The path to the dataset to evaluate the model on.")

    args = args.parse_args()

    # Save templates to local catalog
    catalog_manager = CatalogManager(UnitxtDataConstants.CATALOG_PATH)
    template = catalog_manager.load_from_catalog(args.template_name)

    llm_dataset_loader = DatasetLoader(card=args.card,
                                       template=template,
                                       num_demos=args.num_demos, demos_pool_size=args.demos_pool_size,
                                        system_format=args.system_format, max_train_instances=args.max_train_instances,
                                        template_name=args.template_name)


    llm_dataset = llm_dataset_loader.load()
    llmp = LLMPredictor(args.model_name)

    llm_eval = LLMEvaluator(llmp, llm_dataset)
    results = llm_eval.predict_dataset()
