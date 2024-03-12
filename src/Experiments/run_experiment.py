import argparse
import json

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.ModelsPredictors.LLMPredictor import LLMPredictor
from src.ModelsPredictors.LLMProcessor import LLMProcessor
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    args.add_argument("--card", type=str)
    args.add_argument("--system_format", type=str, default="formats.empty")
    args.add_argument("--max_instances", type=int, default=100)
    args.add_argument('--evaluate_on', nargs='+', default=['train'], help='The data types to evaluate the model on.')
    args.add_argument("--template_num", type=int, default=0)
    args.add_argument("--num_demos", type=int, default=1)
    args.add_argument("--demos_pool_size", type=int, default=10)

    args = args.parse_args()
    template_name = f"template_{args.template_num}"
    catalog_manager = CatalogManager(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH / args.card.split('cards.')[1])
    template = catalog_manager.load_from_catalog(template_name)

    llm_dataset_loader = DatasetLoader(card=args.card,
                                       template=template,
                                       system_format=args.system_format,
                                       num_demos=args.num_demos, demos_pool_size=args.demos_pool_size,
                                       max_instances=args.max_instances,
                                       template_name=template_name)

    llm_dataset = llm_dataset_loader.load()
    # save to json file the params of the experiment and the results of the evaluation
    entry_experiment = {
        "card": args.card,
        "template_name": template_name,
        "model_name": args.model_name,
        "system_format": args.system_format,
        "max_instances": args.max_instances,
        "num_demos": args.num_demos,
        "demos_pool_size": args.demos_pool_size,
        "results": {}
    }
    json_file_name = "experiment_" + args.card + "_" + template_name + ".json"
    results_path = ExperimentConstants.RESULTS_PATH
    results_file_name = results_path / json_file_name
    with open(results_file_name, 'w') as json_file:
        json.dump(entry_experiment, json_file)
    print(f"Results will be saved in {results_file_name}")

    llm_proc = LLMProcessor(args.model_name)
    llm_pred = LLMPredictor(llm_proc)
    results = llm_pred.predict_dataset(llm_dataset, args.evaluate_on, results_file_name=results_file_name)
