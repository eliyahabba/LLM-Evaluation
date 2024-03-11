import argparse

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.DatasetLoader import DatasetLoader
from src.ModelsPredictors.LLMEvaluator import LLMEvaluator
from src.ModelsPredictors.LLMPredictor import LLMPredictor
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    args.add_argument("--card", type=str, default="cards.copa")
    args.add_argument("--num_demos", type=int, default=1)
    args.add_argument("--demos_pool_size", type=int, default=5)
    args.add_argument("--system_format", type=str, default="formats.user_agent")
    args.add_argument("--max_train_instances", type=int, default=5)
    args.add_argument('--evaluate_on', nargs='+', default=['train'], help='The data types to evaluate the model on.')

    args.add_argument("--template_name", type=str, default="template_0")

    args = args.parse_args()

    catalog_manager = CatalogManager(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH)
    template = catalog_manager.load_from_catalog(args.template_name)

    llm_dataset_loader = DatasetLoader(card=args.card,
                                       template=template,
                                       num_demos=args.num_demos, demos_pool_size=args.demos_pool_size,
                                       system_format=args.system_format, max_train_instances=args.max_train_instances,
                                       template_name=args.template_name)

    llm_dataset = llm_dataset_loader.load()
    llmp = LLMPredictor(args.model_name)

    llm_eval = LLMEvaluator(llmp)
    results = llm_eval.predict_dataset(llm_dataset, args.evaluate_on)
