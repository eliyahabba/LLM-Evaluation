from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig


class HellaSwagConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_hellaswag_instructions_standard(self) -> str:
        return f"Pick the best ending to the sentence.\nContext: {{context}}\nChoices:\n{{choices}}\nAnswer:"

    def get_hellaswag_instructions_context(self) -> str:
        return f"Pick the best ending to the context.\nContext: {{context}}\nChoices:\n{{choices}}\nAnswer:"

    def get_hellaswag_instructions_structured(self) -> str:
        return f"Context: [context] Choices: [choices] Answer: [answer]\nContext: {{context}} Choices:\n{{choices}}\nAnswer:"

    def get_hellaswag_instructions_basic(self) -> str:
        return f"Context: {{context}}\n\nChoices: {{choices}}\nAnswer:"

    def get_hellaswag_paraphrase_State_of_What_Art_paper_1(self) -> str:
        return f"Complete the following scenario by selecting the most appropriate ending.\n\nContext: {{context}}\n\nChoices:\n{{choices}}\nAnswer:"

    def get_hellaswag_paraphrase_State_of_What_Art_paper_2(self) -> str:
        return f"Select the most suitable conclusion for the sentence given the context: {{context}}. Here are your options: {{choices}}. Please provide your answer."

    def get_hellaswag_paraphrase_State_of_What_Art_paper_3(self) -> str:
        return f"Choose the most suitable ending for the sentence based on the given context.\nContext: {{context}}\nOptions:\n{{choices}}\nYour Answer:"

    def get_hellaswag_paraphrase_State_of_What_Art_paper_4(self) -> str:
        return f"Select the most suitable conclusion for the sentence.\nGiven Context: {{context}}\nOptions:\n{{choices}}\nResponse:"

    def get_hellaswag_paraphrase_State_of_What_Art_paper_5(self) -> str:
        return f"Choose the most suitable conclusion for the sentence.\nContext: {{context}}\nOptions:\n{{choices}}\nResponse:"

    def get_hellaswag_paraphrase_State_of_What_Art_paper_6(self) -> str:
        return f"Given the context and choices provided, select the most appropriate ending for the sentence. Use the context to understand the situation and choose the option that best completes the sentence. Context: {{context}} Choices: {{choices}} Answer:"

    def get_hellaswag_paraphrase_State_of_What_Art_paper_7(self) -> str:
        return f"Pick the best ending to the sentence based on the given context and choices. Use the context provided to determine the most suitable choice. \nContext: {{context}}\nChoices:\n{{choices}}\nAnswer:"

    def get_hellaswag_paraphrase_State_of_What_Art_paper_8(self) -> str:
        return f"Based on the provided context, choose the most suitable ending for the sentence from the given options. Context: {{context}} Choices: {{choices}} Answer:"

if __name__ == "__main__":
    config = HellaSwagConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
