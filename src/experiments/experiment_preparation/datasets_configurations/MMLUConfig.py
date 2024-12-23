from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig


class MMLUConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_mmlu_instructions_with_topic(self) -> str:
        return f"The following are multiple choice questions (with answers) about {{topic}}.\n\n{{question}}\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_without_topic(self) -> str:
        return f"The following are multiple choice questions (with answers).\n\n{{question}}\n\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_with_topic_helm(self) -> str:
        return f"The following are multiple choice questions (with answers) about {{topic}}.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_without_topic_helm(self) -> str:
        return f"The following are multiple choice questions (with answers).\n\nQuestion: {{question}}\n\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_without_topic_lm_evaluation_harness(self) -> str:
        return f"Question: {{question}}\n\nChoices: {{choices}}\nAnswer:"

    def get_structured_instruction_with_topic(self):
        return f"Topic: {{topic}}\nQuestion: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"

    def get_structured_instruction_without_topic(self):
        return f"Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"

if __name__ == "__main__":
    config = MMLUConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
