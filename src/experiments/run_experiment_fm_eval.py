import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

from termcolor import colored
from unitxt.templates import Template

from src.experiments.data_loading.CatalogManager import CatalogManager
from src.experiments.data_loading.DatasetLoader import DatasetLoader
from src.experiments.models_predictors.LLMPredictor import LLMPredictor
from src.experiments.models_predictors.LLMProcessor import LLMProcessor
from src.utils.Constants import Constants
from src.utils.ReadLLMParams import ReadLLMParams
from src.utils.Utils import Utils
from fm_eval.benchmarks.basic.benchmarks_definitions.utils.benchmark_function import (
    default_system_prompt_and_formatter_mapper,
    get_basic_benchmark_function,
)

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants

@dataclass
class _DefaultGenerationArgs(GenerationArgs):
    max_new_tokens: int = 64
    seed: List[int] = field(default_factory=lambda: [42])
    top_p: List[float] = field(default_factory=lambda: [])
    top_k: List[int] = field(default_factory=lambda: [])
    temperature: List[float] = field(default_factory=lambda: [])
    do_sample: bool = False
    num_beams: int = 1
    stop_sequences: List[List[str]] = field(default_factory=lambda: [["\n\n"]])
    max_predict_samples: int = 20


def get_generation_args(unitxt_args: UnitxtSingleRecipeArgs) -> GenerationArgs:
    return _DefaultGenerationArgs()

class ExperimentRunner:
    """
    This class is responsible for running the experiments.
    """

    def __init__(self, args):
        self.args = args

    def load_template(self, template_num: int, multiple_choice_path) -> \
            Tuple[str, Template]:
        """
        Loads the template from the specified path.
        @return: The template
        """
        template_name = Utils.get_template_name(template_num)
        catalog_manager = CatalogManager(
            Utils.get_card_path(multiple_choice_path, self.args.card))
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
        num_of_shot_str = "zero" if num_demos == 0 else "one" if num_demos == 1 else \
            "two" if num_demos == 2 else "three" if num_demos == 3 else None
        if num_of_shot_str is None:
            raise ValueError(f"num_demos should be between 0 and 2, but it is {num_demos}.")
        self.num_of_shot_icl = f"{num_of_shot_str}_shot"

        system_foramt_name = ExperimentConstants.SYSTEM_FORMATS_NAMES[self.args.system_format]
        results_file_path = (self.args.results_path /
                             Utils.get_model_name(self.args.model_name) /
                             Utils.get_card_name(self.args.card) /
                             self.num_of_shot_icl /
                             system_foramt_name /
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
        if results_file_path.exists() and not results_file_path.stat().st_size == 0:
            # check if the entry_experiment is equal to the one in the file
            with open(results_file_path, 'r') as json_file:
                data = json.load(json_file)
                for key in data:
                    if key != "results":
                        if data[key] != entry_experiment[key]:
                            if key == "max_instances":
                                print(f"max_instances in the current experiment: {entry_experiment[key]}")
                                print(f"max_instances in the previous experiment: {data[key]}")
                            else:
                                # print the key and the value of the key in the current experiment and the previous experiment
                                print(f"{key} in the current experiment: {entry_experiment[key]}")
                                print(f"{key} in the previous experiment: {data[key]}")

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
        llm_proc = LLMProcessor(model_name=self.args.model_name,
                                load_in_4bit=self.args.load_in_4bit,
                                load_in_8bit=self.args.load_in_8bit,
                                trust_remote_code=self.args.trust_remote_code,
                                return_token_type_ids=self.args.not_return_token_type_ids)
        for template_num in range(min_template, max_template):
            start = time.time()
            self.run_single_experiment(llm_proc, template_num)
            end = time.time()
            # print the time of the experiment in minutes (blue color)
            print(colored(f"Time of the experiment: {round((end - start) / 60, 2)} minutes", "blue"))

    def run_single_experiment(self, llm_proc: LLMProcessor, template_num: int) -> None:
        template_name, template = self.load_template(template_num, multiple_choice_path=self.args.multiple_choice_path)
        llm_dataset_loader = DatasetLoader(card=self.args.card,
                                           template=template,
                                           system_format=self.args.system_format,
                                           demos_taken_from=self.args.demos_taken_from,
                                           num_demos=self.args.num_demos, demos_pool_size=self.args.demos_pool_size,
                                           max_instances=self.args.max_instances,
                                           template_name=template_name)
        unitxt_recipe_args = llm_dataset_loader.load()

        entry_experiment = self.create_entry_experiment(template_name)
        results_file_path = self.save_results_to_json(entry_experiment, template_name, self.args.num_demos)

        num_of_possible_answers = 10 if "pro" in self.args.card else 2 if "boolq" in self.args.card else 4
        enumerators = template.enumerator[:num_of_possible_answers]
        if isinstance(enumerators, str):
            # convrte each char tp an element in a list
            enumerators = list(enumerators)
        # for each element in the enumerator list add another token of " " + token
        enumerators_with_space = [f" {enumerator}" for enumerator in enumerators]
        enumerators_with_space2 = [f"{enumerator} " for enumerator in enumerators]
        enumerators_with_dot = [f"{enumerator}." for enumerator in enumerators]
        possible_gt_tokens = enumerators + enumerators_with_space + enumerators_with_dot + enumerators_with_space2

        unitxt_recipe_args_by_groupings: Dict[str, List[UnitxtRecipeArgs]] = {
            "classification": [
                unitxt_recipe_args,
            ],
        }

        get_single_runs_args_list = get_basic_benchmark_function(
            unitxt_recipe_args_by_groupings,
            get_run_generation_args_func=get_generation_args,
            get_train_args_func=None,
            system_prompts_and_formatters_mapper=default_system_prompt_and_formatter_mapper,
        )

        #TODO: Implement the code that runs the experiment




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

    args.add_argument("--predict_prob_of_tokens", default=LLMProcessorConstants.PREDICT_PROB_OF_TOKENS,
                      help="Whether to predict the probability of each token.", action="store_false")
    args.add_argument("--predict_perplexity", default=LLMProcessorConstants.PREDICT_PERPLEXITY,
                      help="Whether to predict the perplexity of the token.", action="store_false")
    args.add_argument("--card", type=str, default="cards.mmlu.anatomy")
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
    # print the args for this experiment: model name, card name, template range, num of demos, demos_pool_size
    print(f"Model name: {args.model_name}, Card name: {args.card}, Template range: {args.template_range}, "
          f"Number of demos: {args.num_demos}, Demos pool size: {args.demos_pool_size}")
    runner.run_experiment()


if __name__ == "__main__":
    # measure the time of the experiment
    start = time.time()
    main()
    end = time.time()
    # print the time of the experiment in minutes (blue color)
    print(colored(f"Total time of the experiment: {round((end - start) / 60, 2)} minutes", "blue"))
