from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig


class BoolQConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_input_format(self):
        return self.get_structured_instruction_text(self.get_context_topic())

    def get_structured_instruction_text(self, context_topic):
        # Provide a default implementation if needed
        return f"Context: [Context]\nQuestion: [question] Choices: [choices] Answer: [answer]\nContext: {{context}}\nQuestion: {{question}} Choices: {{choices}} Answer:"



if __name__ == "__main__":
    config = BoolQConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
