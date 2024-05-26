import sys
from pathlib import Path

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.Visualization.FindCombinations import FindCombinations
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants
from src.utils.MMLUConstants import MMLUConstants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants
BestCombinationsConstants = Constants.BestCombinationsConstants
TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
BEST_COMBINATIONS_PATH = ExperimentConstants.MAIN_RESULTS_PATH / Path(TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME) / f"{ResultConstants.BEST_COMBINATIONS}.csv"

MMLU_SUBCATEGORIES = MMLUConstants.SUBCATEGORIES
MMLU_CATEGORIES = MMLUConstants.SUBCATEGORIES_TO_CATEGORIES
BestOrWorst = Constants.BestOrWorst

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=BEST_COMBINATIONS_PATH,
                        help="Path to the best combinations file")
    args = parser.parse_args()
    best_combinations_displayer = FindCombinations(BestOrWorst.BEST, args.file_path, ConfigParams.override_options)
    best_combinations_displayer.evaluate()
