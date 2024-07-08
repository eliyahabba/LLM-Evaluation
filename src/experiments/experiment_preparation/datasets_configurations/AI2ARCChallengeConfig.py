from src.experiments.experiment_preparation.configuration_generation.BaseDatasetConfig import BaseDatasetConfig


class AI2ARCChallengeConfig(BaseDatasetConfig):
    def __init__(self):
        super().__init__()  # Initialize base class properties
        self.enumerator = "capitals"  # Example of overriding a base class property

    def get_input_format(self):
        return ("Start with the topic, then detail the question, choices, and the correct answer:\n"
                "Topic: {topic}\nQuestion: {question}\nChoices: {choices}\nCorrect Answer:")


if __name__ == "__main__":
    config = AI2ARCChallengeConfig()
    config_dict = config.to_dict()
    print(config_dict)