import argparse
import json
from pathlib import Path
from typing import Tuple

from termcolor import colored
from unitxt.templates import Template

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.ModelsPredictors.LLMPredictor import LLMPredictor
from src.ModelsPredictors.LLMProcessor import LLMProcessor
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class ExperimentRunner:
    """
    This class is responsible for running the experiments.
    """

    def __init__(self, args):
        self.args = args

    def load_template(self) -> Tuple[str, Template]:
        """
        Loads the template from the specified path.
        @return: The template
        """
        template_name = Utils.get_template_name(self.args.template_num)
        catalog_manager = CatalogManager(
            Utils.get_card_path(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH, self.args.card))
        template = catalog_manager.load_from_catalog(template_name)
        return template_name, template

    def create_entry_experiment(self, template_name: str) -> dict:
        """
        Creates the entry for the experiment.
        @param template_name: The name of the template.
        @return: The entry for the experiment.
        """
        entry_experiment = {
            "card": self.args.card,
            "template_name": template_name,
            "model_name": self.args.model_name,
            "system_format": self.args.system_format,
            "max_instances": self.args.max_instances,
            "num_demos": self.args.num_demos,
            "demos_pool_size": self.args.demos_pool_size,
            "results": {}
        }
        return entry_experiment

    def get_result_file_path(self, template_name: str, num_demos: int) -> Path:
        """
        Returns the path to the results file.
        @param template_name: The name of the template.
        @param num_demos: The number of demos (The number of demonstrations for in-context learning).
        @return: The path to the results file.
        """
        json_file_name = "experiment_" + template_name + ".json"
        num_of_shot_str = "zero" if num_demos == 0 else "one" if num_demos == 1 else "two" if num_demos == 2 \
            else None
        if num_of_shot_str is None:
            raise ValueError(f"num_demos should be between 0 and 2, but it is {num_demos}.")
        num_of_shot_icl = f"{num_of_shot_str}_shot"
        results_path = ExperimentConstants.RESULTS_PATH
        results_file_path = results_path / self.args.card.split('cards.')[1] / num_of_shot_icl / json_file_name
        results_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved in {results_file_path}")
        return results_file_path

    def save_results_to_json(self, entry_experiment: dict, template_name: str, num_demos: int) -> Path:
        """
        Saves the results to a json file.
        @param entry_experiment: The entry for the experiment.
        @param template_name: The name of the template.
        @param num_demos: The number of demos (The number of demonstrations for in-context learning).
        @return: The path to the results file.
        """
        results_file_path = self.get_result_file_path(template_name, num_demos)
        if results_file_path.exists():
            # check if the entry_experiment is equal to the one in the file
            with open(results_file_path, 'r') as json_file:
                data = json.load(json_file)
                for key in data:
                    if key != "results":
                        if data[key] != entry_experiment[key]:
                            raise ValueError(f"The metadata of the experiment in {results_file_path} "
                                             f"is different from the one in the current experiment.")
                # print blue message
            print(colored(f"Results already exist in {results_file_path}", "blue"))
            return results_file_path
        with open(results_file_path, 'w') as json_file:
            json.dump(entry_experiment, json_file)
        return results_file_path

    def run_experiment(self) -> list:
        """
        Runs the experiment.
        @return: The results of the experiment.
        """
        template_name, template = self.load_template()
        llm_dataset_loader = DatasetLoader(card=self.args.card,
                                           template=template,
                                           system_format=self.args.system_format,
                                           num_demos=self.args.num_demos, demos_pool_size=self.args.demos_pool_size,
                                           max_instances=self.args.max_instances,
                                           template_name=template_name)
        llm_dataset = llm_dataset_loader.load()

        entry_experiment = self.create_entry_experiment(template_name)
        results_file_path = self.save_results_to_json(entry_experiment, template_name, self.args.num_demos)

        llm_proc = LLMProcessor(self.args.model_name)
        llm_pred = LLMPredictor(llm_proc)
        results = llm_pred.predict_dataset(llm_dataset, self.args.evaluate_on, results_file_path=results_file_path)
        return results


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--card", type=str)
    args.add_argument("--model_name", type=str, default=Constants.ExperimentConstants.MODEL_NAME)
    args.add_argument("--system_format", type=str, default=Constants.ExperimentConstants.SYSTEM_FORMATS)
    args.add_argument("--max_instances", type=int, default=Constants.ExperimentConstants.MAX_INSTANCES)
    args.add_argument('--evaluate_on', nargs='+', default=Constants.ExperimentConstants.EVALUATE_ON,
                      help='The data types to evaluate the model on.')
    args.add_argument("--template_num", type=int, default=Constants.ExperimentConstants.TEMPLATE_NUM)
    args.add_argument("--num_demos", type=int, default=Constants.ExperimentConstants.NUM_DEMOS)
    args.add_argument("--demos_pool_size", type=int, default=Constants.ExperimentConstants.DEMOS_POOL_SIZE)

    args = args.parse_args()

    runner = ExperimentRunner(args)
    runner.run_experiment()


if __name__ == "__main__":
    main()
