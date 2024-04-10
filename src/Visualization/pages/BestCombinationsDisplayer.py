import pandas as pd
import streamlit as st

from src.utils.Constants import Constants

ExperimentConstants = Constants.ExperimentConstants
ResultConstants = Constants.ResultConstants

file_path = ExperimentConstants.STRUCTURED_INPUT_FOLDER_PATH / f"{ResultConstants.BEST_COMBINATIONS}.csv"


class BestCombinationsDisplayer:
    def __init__(self):
        self.best_combinations = pd.read_csv(file_path)

    def display(self):
        st.title("Best Combinations")
        # add subtitle that table displays the best combinations for each model and dataset on all the possible axes
        st.write("The table below displays the best combinations for each model and dataset on all the possible axes.")
        # write the table
        st.write(self.best_combinations)

if __name__ == "__main__":
    BestCombinationsDisplayer().display()