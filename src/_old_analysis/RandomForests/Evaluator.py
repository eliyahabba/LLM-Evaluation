import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, y_test: pd.Series, predictions: np.ndarray):
        self.y_test = y_test
        self.predictions = predictions

    def round_digits(self, x, k=2):
        """trunc x to k digits"""
        return np.round(x, k)
    def evaluate(self):
        metrics = {}
        metrics["accuracy"] = self.evaluate_accuracy(self.y_test, self.predictions)
        metrics["distance"] = self.evaluate_distance(self.y_test, self.predictions)
        metrics["by_class"] = self.evaluate_by_class(self.y_test, self.predictions)
        return metrics

    def evaluate_accuracy(self, y_test: pd.Series
                          , predictions: np.ndarray):
        """Evaluate the model using accuracy."""
        return (y_test == predictions).mean().round(2)

    def letter_to_index(self, letter):
        """Convert a letter to its corresponding index (A=1, B=2, ..., Z=26)."""
        return ord(letter.upper()) - ord('A') + 1

    def evaluate_distance(self, y_test: pd.Series, predictions: np.ndarray):
        """Evaluate the model with a custom metric that considers the proximity of the predicted letter to the true letter."""
        # Convert letters to indices
        y_test = y_test.to_numpy()
        y_test_indices = np.vectorize(self.letter_to_index)(y_test)
        predictions_indices = np.vectorize(self.letter_to_index)(predictions)

        # Calculate the absolute differences
        differences = np.abs(y_test_indices - predictions_indices)

        # Define a custom weighting system if needed
        # For example, differences of 1 might be less penalized
        # Here, a simple inverse of difference is used as weight; you can modify it as needed
        non_zero_indices = differences != 0
        weights = np.zeros_like(differences, dtype=float)
        weights[non_zero_indices] = 1 / differences[non_zero_indices]
        # Calculate weighted accuracy
        weighted_accuracy = weights.mean().round(2)

        return weighted_accuracy

    def evaluate_by_class(self, y_test: pd.Series, predictions: np.ndarray):
        """Evaluate the model by class."""
        class_metrics = {}
        for class_name in y_test.unique():
            class_indices = y_test == class_name
            # class_metrics[class_name] = { "accuracy": self.evaluate_accuracy(y_test[class_indices], predictions[class_indices]) }
            # calculate precision, recall, f1-score, etc.
            recall = np.sum(predictions[class_indices] == class_name) / np.sum(y_test == class_name)
            precision = 0.0 if np.sum(predictions == class_name) ==0 else np.sum(predictions[class_indices] == class_name) / np.sum(predictions == class_name)
            class_metrics[class_name] = {}
            class_metrics[class_name]["recall"] = round(recall, 2)
            class_metrics[class_name]["precision"] = round(precision, 2)
            class_metrics[class_name]["f1-score"] = 0 if precision + recall== 0 else round((2 * (precision * recall) / (precision + recall)), 2)
        return class_metrics

if __name__ == "__main__":
    y_test = pd.Series(["A", "B", "C", "D", "E"])
    predictions = np.array(["A", "B", "C", "D", "F"])

    evaluator = Evaluator(y_test, predictions)
    metrics = evaluator.evaluate()
    print("Metrics:", metrics)