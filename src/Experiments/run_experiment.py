import argparse
import json
import time
from pathlib import Path
from typing import Tuple

from termcolor import colored
from unitxt.templates import Template

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.ModelsPredictors.LLMPredictor import LLMPredictor
from src.ModelsPredictors.LLMProcessor2 import LLMProcessor
from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
RESULTS_PATH = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH


class ExperimentRunner:
    """
    This class is responsible for running the experiments.
    """

    def __init__(self, args):
        self.args = args

    def load_template(self, template_num: int) -> Tuple[str, Template]:
        """
        Loads the template from the specified path.
        @return: The template
        """
        template_name = Utils.get_template_name(template_num)
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
            "system_format": ExperimentConstants.SYSTEM_FORMATS_NAMES[self.args.system_format],
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

        system_foramt_name = ExperimentConstants.SYSTEM_FORMATS_NAMES[self.args.system_format]
        model_name = self.args.model_name.split('/')[-1]
        results_file_path = (RESULTS_PATH /
                             model_name /
                             self.args.card.split('cards.')[1] / num_of_shot_icl / system_foramt_name / \
                             json_file_name)
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

    def run_experiment(self) -> None:
        """
        Runs the experiment.
        @return: The results of the experiment.
        """
        min_template, max_template = self.args.template_range
        llm_proc = LLMProcessor(self.args.model_name,
                                self.args.not_load_in_4bit, self.args.not_load_in_8bit,
                                self.args.trust_remote_code, self.args.not_return_token_type_ids)
        for template_num in range(min_template, max_template + 1):
            start = time.time()
            self.run_single_experiment(llm_proc, template_num)
            end = time.time()
            # print the time of the experiment in minutes (blue color)
            print(colored(f"Time of the experiment: {round((end - start) / 60, 2)} minutes", "blue"))

    def run_single_experiment(self, llm_proc: LLMProcessor, template_num: int) -> None:
        template_name, template = self.load_template(template_num)
        llm_dataset_loader = DatasetLoader(card=self.args.card,
                                           template=template,
                                           system_format=self.args.system_format,
                                           num_demos=self.args.num_demos, demos_pool_size=self.args.demos_pool_size,
                                           max_instances=self.args.max_instances,
                                           template_name=template_name)
        llm_dataset = llm_dataset_loader.load()

        entry_experiment = self.create_entry_experiment(template_name)
        results_file_path = self.save_results_to_json(entry_experiment, template_name, self.args.num_demos)

        llm_pred = LLMPredictor(llm_proc, batch_size=self.args.batch_size)
        llm_pred.predict_dataset(llm_dataset, self.args.evaluate_on, results_file_path=results_file_path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--card", type=str, default="cards.sciq")
    args.add_argument("--model_name", type=str, default=LLMProcessorConstants.PHI_MODEL)
    args.add_argument("--not_load_in_4bit", action="store_false", default=LLMProcessorConstants.LOAD_IN_4BIT,
                      help="True if the model should be loaded in 4-bit.")
    args.add_argument("--not_load_in_8bit", action="store_false", default=LLMProcessorConstants.LOAD_IN_8BIT,
                      help="True if the model should be loaded in 8-bit.")
    args.add_argument("--trust_remote_code", action="store_true", default=LLMProcessorConstants.TRUST_REMOTE_CODE,
                      help="True if the model should trust remote code.")
    args.add_argument("--not_return_token_type_ids", action="store_false",
                      default=LLMProcessorConstants.RETURN_TOKEN_TYPE_IDS,
                      help="True if the model should not return token type ids.")
    args.add_argument("--system_format_index", type=int, default=ExperimentConstants.SYSTEM_FORMAT_INDEX)

    args.add_argument("--batch_size", type=int, default=ExperimentConstants.BATCH_SIZE, help="The batch size.")
    args.add_argument("--max_instances", type=int, default=ExperimentConstants.MAX_INSTANCES)
    args.add_argument('--evaluate_on', nargs='+', default=ExperimentConstants.EVALUATE_ON,
                      help='The data types to evaluate the model on.')
    args.add_argument("--num_demos", type=int, default=ExperimentConstants.NUM_DEMOS)
    args.add_argument("--demos_pool_size", type=int, default=ExperimentConstants.DEMOS_POOL_SIZE)
    # args.add_argument("--template_num", type=int, default=ExperimentConstants.TEMPLATE_NUM)
    # # option to give a range of templates to run the experiment on (e.g. 1 10). with 2 parameters min and max template
    args.add_argument("--template_range", nargs=2, type=int, default=ExperimentConstants.TEMPLATES_RANGE,
                      help="Specify the range of templates to run the experiment on (e.g., 1 10).")
    # add param
    args = args.parse_args()
    # add the syte  format to the args
    args.system_format = ExperimentConstants.SYSTEM_FORMATS[args.system_format_index]
    # map between the model name to the real model name from the constants
    args.model_name = LLMProcessorConstants.MODEL_NAMES[args.model_name]
    if args.card.split("cards.")[1] == Constants.DatasetsConstants.MMLU_GENERAL:
        # run on all the MMLU datasets with a loop
        for card in Constants.DatasetsConstants.MMLU_DATASETS_SAMPLE:
            args.card = f"cards.{card}"
            runner = ExperimentRunner(args)
            runner.run_experiment()
    else:
        runner = ExperimentRunner(args)
        runner.run_experiment()


if __name__ == "__main__":
    # measure the time of the experiment
    main()
