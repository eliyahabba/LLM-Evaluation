import argparse
from pathlib import Path

import pandas as pd

from src.RandomForests.Constants import RandomForestsConstants
from src.RandomForests.GroupPredictor import GroupPredictor


class RandomForest:
    def __init__(self, configurations_data_path: Path):
        self.configurations_data_path: Path = configurations_data_path
        self.configurations_data: pd.DataFrame = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.configurations_data_path)
        # Drop rows if Group is NaN
        df = df.dropna(subset=[RandomForestsConstants.GROUP])
        return df

    def create_model(self):
        self._load_data()
        self.predictor = GroupPredictor()

    def train(self):
        X_train, X_test, y_train, y_test = self.predictor.load_and_split_data(self.configurations_data)
        self.predictor.train(X_train, y_train)

    def predict(self):
        # New data for prediction
        new_data = [
            {"model": "Llama-2-7b-chat-hf", "enumerator": "lowercase", "choices_separator": ", ",
             "shuffle_choices": True},
            {"model": "Llama-2-7b-chat-hf", "enumerator": "roman", "choices_separator": " OR ",
             "shuffle_choices": False}
        ]
        new_df = self.predictor.prepare_data(new_data)
        predictions = self.predictor.predict(new_df)
        print("Predictions:", predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations_data_path", type=str, default=RandomForestsConstants.CONFIGURATIONS_DATA_PATH,
                        help="Path to the best combinations file")
    args = parser.parse_args()
    rf = RandomForest(args.configurations_data_path)
    rf.create_model()
    rf.train()
    rf.predict()
