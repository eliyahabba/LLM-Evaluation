from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig


class HellaSwagConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_hellaswag_instructions_standard(self) -> str:
        return f"Pick the best ending to the sentence.\nContext: {{context}}\nChoices:\n{{choices}}\nAnswer:"

    def get_hellaswag_instructions_sentence(self) -> str:
        return f"Pick the best ending to the Sentence.\nSentence: {{context}}\nChoices:\n{{choices}}\nAnswer:"

    def get_hellaswag_instructions_structured(self) -> str:
        return f"Context: [context] Choices: [choices] Answer: [answer]\nContext: {{context}} Choices:\n{{choices}}\nAnswer:"

    def get_hellaswag_instructions_basic(self) -> str:
        return f"Context: {{context}}\n\nChoices: {{choices}}\nAnswer:"

    def get_hellaswag_paraphrase1(self) -> str:
        return f"Complete the following scenario by selecting the most appropriate ending.\n\nContext: {{context}}\n\nChoices:\n{{choices}}\nAnswer:"


if __name__ == "__main__":
    config = HellaSwagConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
