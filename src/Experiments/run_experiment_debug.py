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
from src.ModelsPredictors.LLMProcessor import LLMProcessor
from src.utils.Constants import Constants
from src.utils.ReadLLMParams import ReadLLMParams
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants


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

        system_foramt = Utils.get_system_format_class(self.args.system_format)

        results_path = ExperimentConstants.MAIN_RESULTS_PATH / Path(
            TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME)
        results_file_path = results_path / self.args.card.split('cards.')[1] / num_of_shot_icl / system_foramt / \
                            json_file_name
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
        for template_num in range(min_template, max_template + 1):
            start = time.time()
            self.run_single_experiment(template_num)
            end = time.time()
            # print the time of the experiment in minutes (blue color)
            print(colored(f"Time of the experiment: {round((end - start) / 60, 2)} minutes", "blue"))

    def run_single_experiment(self, template_num: int) -> None:
        template_name, template = self.load_template(template_num)
        llm_dataset_loader = DatasetLoader(card=self.args.card,
                                           template=template,
                                           system_format=self.args.system_format,
                                           demos_taken_from=self.args.demos_taken_from,
                                           num_demos=self.args.num_demos, demos_pool_size=self.args.demos_pool_size,
                                           max_instances=self.args.max_instances,
                                           template_name=template_name)
        llm_dataset = llm_dataset_loader.load()

        entry_experiment = self.create_entry_experiment(template_name)
        results_file_path = self.save_results_to_json(entry_experiment, template_name, self.args.num_demos)

        llm_proc = LLMProcessor(self.args.model_name, self.args.load_in_4bit, self.args.load_in_8bit)
        llm_pred = LLMPredictor(llm_proc)

        enumerators = template.enumerator[:4]
        if isinstance(enumerators, str):
            # convrte each char tp an element in a list
            enumerators = list(enumerators)
        # for each element in the enumerator list add another token of " " + token
        enumerators_with_space = [f" {enumerator}" for enumerator in enumerators]
        possible_gt_tokens = enumerators + enumerators_with_space

        llm_pred.predict_dataset(llm_dataset, self.args.evaluate_on, results_file_path=results_file_path)


def parser_bit_precision(args: argparse.Namespace) -> Tuple[bool, bool]:
    # Resolve conflicts and decide final settings
    if args.not_load_in_8bit and args.not_load_in_4bit:
        load_in_4bit = False
        load_in_8bit = False
    elif args.load_in_4bit:
        load_in_4bit = True
        load_in_8bit = False
    elif args.load_in_8bit:
        load_in_4bit = False
        load_in_8bit = True
    else:
        load_in_4bit = False
        load_in_8bit = False

    return load_in_4bit, load_in_8bit


def main():
    args = argparse.ArgumentParser()
    args = ReadLLMParams.read_llm_params(args)

    args.add_argument("--card", type=str, default="cards.mmlu.college_medicine")
    args.add_argument("--system_format_index", type=int, default=ExperimentConstants.SYSTEM_FORMAT_INDEX)
    args.add_argument("--batch_size", type=int, default=ExperimentConstants.BATCH_SIZE, help="The batch size.")
    args.add_argument("--max_instances", type=int, default=ExperimentConstants.MAX_INSTANCES)
    args.add_argument('--evaluate_on', nargs='+', default=ExperimentConstants.EVALUATE_ON_INFERENCE,
                      help='The data types to evaluate the model on.')
    args.add_argument("--demos_taken_from", type=str, default=ExperimentConstants.DEMOS_TAKEN_FROM)
    args.add_argument("--num_demos", type=int, default=ExperimentConstants.NUM_DEMOS)
    args.add_argument("--demos_pool_size", type=int, default=ExperimentConstants.DEMOS_POOL_SIZE)
    # args.add_argument("--template_num", type=int, default=ExperimentConstants.TEMPLATE_NUM)
    # # option to give a range of templates to run the experiment on (e.g. 1 10). with 2 parameters min and max template
    args.add_argument("--template_range", nargs=2, type=int, default=ExperimentConstants.TEMPLATES_RANGE,
                      help="Specify the range of templates to run the experiment on (e.g., 1 10).")
    # add param
    args = args.parse_args()
    # check if load_in_4bit or not_load_in_4bit
    args.load_in_4bit, args.load_in_8bit = parser_bit_precision(args)

    # add the syte  format to the args
    args.system_format = ExperimentConstants.SYSTEM_FORMATS[args.system_format_index]
    # map between the model name to the real model name from the constants
    args.model_name = LLMProcessorConstants.MODEL_NAMES[args.model_name]
    args.multiple_choice_path = TemplatesGeneratorConstants.DATA_PATH / Path(args.multiple_choice_name)
    args.results_path = ExperimentConstants.MAIN_RESULTS_PATH / Path(args.multiple_choice_name)
    runner = ExperimentRunner(args)
    runner.run_experiment()


if __name__ == "__main__":
    # measure the time of the experiment
    start = time.time()
    main()
    end = time.time()
    # print the time of the experiment in minutes (blue color)
    print(colored(f"Total time of the experiment: {round((end - start) / 60, 2)} minutes", "blue"))
