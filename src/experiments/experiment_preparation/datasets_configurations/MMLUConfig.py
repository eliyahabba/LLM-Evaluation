from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig


class MMLUConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_mmlu_instructions_with_topic(self, topic: str) -> str:
        return f"The following are multiple choice questions (with answers) about {topic}.\n"

    def get_mmlu_instructions_without_topic(self) -> str:
        return "The following are multiple choice questions (with answers).\n"

    def get_structured_instruction_text_with_topic(self):
        # Provide a default implementation if needed
        return f"Topic: {{topic}}\nQuestion: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"

    def get_structured_instruction_text_without_topic(self):
        # Provide a default implementation if needed
        return f"Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"


if __name__ == "__main__":
    config = MMLUConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
