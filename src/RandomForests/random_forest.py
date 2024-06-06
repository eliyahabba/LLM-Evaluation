import argparse
from pathlib import Path

import pandas as pd

from src.RandomForests.Constants import RandomForestsConstants
from src.RandomForests.GroupPredictor import GroupPredictor


class RandomForest:
    def __init__(self, configurations_data_path: Path = RandomForestsConstants.CONFIGURATIONS_DATA_PATH,
                 feature_columns: list = None):
        self.configurations_data = None
        self.predictor = None
        self.configurations_data_path: Path = configurations_data_path
        self.feature_columns = feature_columns

    def load_data(self, model: str = None):
        df = self._read_data()
        if model:
            df = df[df["model"] == model]
        self.configurations_data = df

    def _read_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.configurations_data_path)
        return df

    def _process_data(self, df: pd.DataFrame, target_column: str = RandomForestsConstants.GROUP
                      ) -> pd.DataFrame:
        # Drop rows if Group is NaN
        # take only the first feature columns
        if self.feature_columns:
            # remove all columns that are not in feature_columns list (and not in target_column)
            df = df[self.feature_columns]
        return df

    def create_model(self):
        self.predictor = GroupPredictor(feature_columns=self.feature_columns)

    def split_data(self, split_column_name: str = RandomForestsConstants.CATEGORY, test_column_values: list = None):
        self.configurations_data = self.configurations_data.dropna(subset=[RandomForestsConstants.GROUP])
        X_train, X_test, y_train, y_test = self.predictor.split_data(self.configurations_data)
        X_train, X_test, y_train, y_test = self.split_data_by_column(self.configurations_data, split_column_name,
                                                                     test_column_values)
        X_train = self._process_data(X_train)
        X_test = self._process_data(X_test)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        self.predictor.train(X_train, y_train)

    def predict(self, X_test):
        new_df = self.predictor.prepare_data(X_test)
        predictions = self.predictor.predict(new_df)
        return predictions

    def evaluate(self, y_test, predictions, print_metrics: bool = True):
        metrics = self.predictor.evaluate(y_test, predictions)
        if print_metrics:
            print(metrics)
        return metrics

    def split_data_by_column(self, data, column_name, test_column_values: list = None):
        # Split data into training and testing datasets based on the column_name (so the Xtrain and Xtest will separate
        # the data based on the column_name)
        if test_column_values:
            X_train = data[~data[column_name].isin(test_column_values)]
            X_test = data[data[column_name].isin(test_column_values)]
            y_train = X_train.pop(RandomForestsConstants.GROUP)
            y_test = X_test.pop(RandomForestsConstants.GROUP)
            return X_train, X_test, y_train, y_test
        column_values = data[column_name].unique()
        X_train = data[data[column_name].isin(column_values[:-1])]
        X_test = data[data[column_name] == column_values[-1]]
        y_train = X_train.pop(RandomForestsConstants.GROUP)
        y_test = X_test.pop(RandomForestsConstants.GROUP)
        return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations_data_path", type=str, default=RandomForestsConstants.CONFIGURATIONS_DATA_PATH,
                        help="Path to the best combinations file")

    args = parser.parse_args()
    rf = RandomForest(configurations_data_path=args.configurations_data_path,
                      feature_columns=["dataset", "Sub_Category", "Category", "enumerator", "choices_separator",
                                       "shuffle_choices"])
    rf.load_data(model="Llama-2-7b-chat-hf")
    rf.create_model()
    X_train, X_test, y_train, y_test = rf.split_data(split_column_name=RandomForestsConstants.CATEGORY)
    rf.train(X_train, y_train)
    predictions = rf.predict(X_test)
    print("Test set")
    metrics = rf.evaluate(y_test, predictions, print_metrics=True)

    print("Train set")
    predictions = rf.predict(X_train)
    metrics = rf.evaluate(y_train, predictions, print_metrics=True)


if __name__ == "__main__":
    main()
