

# Usage
from src.Experiments.GroupPredictor import GroupPredictor

if __name__ == "__main__":
    data = [
        {"model": "Llama-2-7b-chat-hf", "enumerator": "capitals", "choices_separator": "\n", "shuffle_choices": False,
         "group": "A"},
        {"model": "Llama-2-7b-chat-hf", "enumerator": "numbers", "choices_separator": "; ", "shuffle_choices": True,
         "group": "B"},
        # Add more data as needed
    ]

    predictor = GroupPredictor()
    X_train, X_test, y_train, y_test = predictor.load_and_split_data(data)
    predictor.train(X_train, y_train)

    # New data for prediction
    new_data = [
        {"model": "Llama-2-7b-chat-hf", "enumerator": "lowercase", "choices_separator": ", ", "shuffle_choices": True},
        {"model": "Llama-2-7b-chat-hf", "enumerator": "roman", "choices_separator": " OR ", "shuffle_choices": False}
    ]
    new_df = predictor.prepare_data(new_data)
    predictions = predictor.predict(new_df)
    print("Predictions:", predictions)
