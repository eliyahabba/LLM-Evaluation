from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig


class AI2ARCChallengeConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_input_format(self):
        return self.get_structured_instruction_text(self.get_context_topic())

    def get_structured_instruction_text(self, context_topic):
        # Provide a default implementation if needed
        return f"{context_topic}Question: [question] Choices: [choices] Answer: [Answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"

    def get_context_topic(self):
        return "Topic: {topic}\n"


if __name__ == "__main__":
    config = AI2ARCChallengeConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
