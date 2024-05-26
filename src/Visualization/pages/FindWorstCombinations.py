import sys
from pathlib import Path

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.Visualization.FindCombinations import FindCombinations
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants

BestOrWorst = Constants.BestOrWorst

if __name__ == "__main__":
    best_combinations_displayer = FindCombinations(BestOrWorst.WORST, ConfigParams.override_options)
    best_combinations_displayer.evaluate()
