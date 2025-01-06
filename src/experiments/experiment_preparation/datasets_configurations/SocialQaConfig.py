from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig


class SocialQaConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_mmlu_instructions_with_topic(self) -> str:
        return f"The following are multiple choice questions (with answers) about common sense social interactions.\n\n{{question}}\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_without_topic(self) -> str:
        return f"The following are multiple choice questions (with answers).\n\n{{question}}\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_with_topic_helm(self) -> str:
        return f"The following are multiple choice questions (with answers) about common sense social interactions.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_without_topic_helm(self) -> str:
        return f"The following are multiple choice questions (with answers).\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"

    def get_mmlu_instructions_without_topic_lm_evaluation_harness(self) -> str:
        return f"Question: {{question}}\n\nChoices: {{choices}}\nAnswer:"

    def get_structured_instruction_with_topic(self):
        return f"Topic: common sense social interactions\nQuestion: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"

    def get_structured_instruction_without_topic(self):
        return f"Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"

    def get_mmlu_instructions_with_topic_and_cot(self):
        return (f"The following are multiple choice questions (with answers) about common sense social interactions. Think step by"
                f" step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n")

    def get_please_simple_prompt_ProSA_paper(self) -> str:
        return f"Please answer the following question:\n{{question}}\n{{choices}}\nAnswer:"

    def get_please_letter_prompt_ProSA_paper(self) -> str:
        return f"Please answer the following question:\n{{question}}\n{{choices}}\nAnswer the question by replying {{options}}."

    def get_could_you_prompt_ProSA_paper(self) -> str:
        return f"Could you provide a response to the following question:\n{{question}}\n{{choices}}\nAnswer:"

    def get_here_prompt_State_of_What_Art_paper(self) -> str:
        return f"Here are some multiple choice questions along with their answers about common sense social interactions.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"

    def get_below_prompt_State_of_What_Art_paper(self) -> str:
        return f"Below are multiple-choice questions related to common sense social interactions, each followed by their respective answers.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"

    def get_below_please_prompt_State_of_What_Art_paper(self) -> str:
        return f"Below are multiple-choice questions related to common sense social interactions. Please provide the correct answer for each question.\n\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"



if __name__ == "__main__":
    config = SocialQaConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
