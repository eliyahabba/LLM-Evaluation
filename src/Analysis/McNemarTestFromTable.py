import argparse
import itertools

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm

from src.utils.Constants import Constants
from src.utils.Utils import Utils

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants
DatasetsConstants = Constants.DatasetsConstants
McNemarTestConstants = Constants.McNemarTestConstants
ResultConstants = Constants.ResultConstants
contingency_table_FOLDER = "structured_input"
SHOT = "zero_shot"
FORMAT = "empty_system_format"
TRAIN_OR_TEST_TYPE = "test"

class McNemarTestFromTable:
    """
    Class to perform McNemar test on the contingency_table provided as a contingency table.
    """

    def __init__(self):
        pass

    @staticmethod
    def perform_mcnemar_test_from_table(results: pd.DataFrame) -> pd.DataFrame:
        """
        Perform McNemar test on the given contingency table.
        @param contingency_table: Contingency table containing the contingency_table
        @return: DataFrame containing McNemar test contingency_table
        """
        # take all the pairs of columns and for each pair, create a contingency table
        column_pairs = itertools.combinations(results.columns, 2)
    
        df = pd.DataFrame(columns=results.columns)
        # add rows index with the same index as the columns
        for col in results.columns:
            df.loc[col] = "None"
        df.set_index(results.columns, inplace=True)
        # create this to be the index of the dataframe
        for column1, column2 in column_pairs:
            contingency_table = pd.crosstab(results[column1], results[column2])
            # Perform McNemar test
            result = mcnemar(contingency_table)
            df.at[column1, column2] = f"{result.statistic}, {result.pvalue}"
            df.at[column2, column1] = f"{result.statistic}, {result.pvalue}"
    
        # if result.pvalue < McNemarTestConstants.ALPHA:
        #     # print(f"contingency_table for {column1} and {column2} are significantly different.")
        #     # print("McNemar Test Statistic:", result.statistic)
        #     # print("McNemar Test P-value:", result.pvalue)
        #     test_contingency_table.append((column1, column2, result.statistic, result.pvalue))
        return df

    @staticmethod
    def perform_cochrans_q_test_from_table(results: pd.DataFrame)->statsmodels.stats.contingency_tables:
        """
        Perform Cochran's Q test on the given contingency table.
        @param contingency_table: Contingency table containing the contingency_table
        """
        # Perform Cochran's Q test
        result = cochrans_q(results)
        return result
