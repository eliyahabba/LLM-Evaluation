import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.RandomForests.Constants import RandomForestsConstants


class GroupPredictor:
    def __init__(self, random_state=42):
        """Initialize the predictor with a random state for reproducibility."""
        self.random_state = random_state
        self.model = self._create_pipeline()

    def _create_pipeline(self):
        """Create a pipeline that includes one-hot encoding and a random forest classifier."""
        categorical_features = ['enumerator', 'choices_separator', 'shuffle_choices', "model"]
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ])
        rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestClassifier(random_state=self.random_state))])
        return rf_pipeline

    def train(self, X_train, y_train):
        """Train the model on training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict using the trained model."""
        return self.model.predict(X)

    def prepare_data(self, data):
        """Prepare the data by converting it from a list of dicts to a DataFrame."""
        return pd.DataFrame(data)

    def load_and_split_data(self, data):
        """Load data, prepare it, and split it into training and testing datasets."""
        df = self.prepare_data(data)
        X = df.drop(RandomForestsConstants.GROUP, axis=1)
        y = df[RandomForestsConstants.GROUP]
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

