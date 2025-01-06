from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig, \
    PromptInstruction


class MMLUConfig(BaseDatasetConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def get_mmlu_instructions_with_topic(self) -> PromptInstruction:
        return PromptInstruction(name="MultipleChoiceTemplatesInstructionsWithTopic",
                                 text=f"The following are multiple choice questions (with answers) about {{topic}}.\n\n{{question}}\n{{choices}}\nAnswer:")

    def get_mmlu_instructions_without_topic(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsWithoutTopic"
        prompt_text = f"The following are multiple choice questions (with answers).\n\n{{question}}\n\n{{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_mmlu_instructions_without_topic_fixed(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsWithoutTopicFixed"
        prompt_text = f"The following are multiple choice questions (with answers).\n\n{{question}}\n{{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_mmlu_instructions_with_topic_helm(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsWithTopicHelm"
        prompt_text = f"The following are multiple choice questions (with answers) about {{topic}}.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_mmlu_instructions_without_topic_helm(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsWithoutTopicHelm"
        prompt_text = f"The following are multiple choice questions (with answers).\n\nQuestion: {{question}}\n\n{{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_mmlu_instructions_without_topic_helm_fixed(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed"
        prompt_text = f"The following are multiple choice questions (with answers).\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_mmlu_instructions_without_topic_lm_evaluation_harness(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsWithoutTopicHarness"
        prompt_text = f"Question: {{question}}\n\nChoices: {{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_structured_instruction_with_topic(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesStructuredWithTopic"
        prompt_text = f"Topic: {{topic}}\nQuestion: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"
        return prompt_name, prompt_text

    def get_structured_instruction_without_topic(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesStructuredWithoutTopic"
        prompt_text = f"Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"
        return prompt_name, prompt_text

    def get_mmlu_instructions_with_topic_and_cot(self) -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsWithTopicAndCoT"
        prompt_text = (f"The following are multiple choice questions (with answers) about {{topic}}. Think step by"
                       f" step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n")
        return prompt_name, prompt_text

    def get_please_simple_prompt_ProSA_paper(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsProSASimple"
        prompt_text = f"Please answer the following question:\n{{question}}\n{{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_please_letter_prompt_ProSA_paper(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsProSALetter"
        prompt_text = f"Please answer the following question:\n{{question}}\n{{choices}}\nAnswer the question by replying {{options}}."
        return prompt_name, prompt_text

    def get_could_you_prompt_ProSA_paper(self) -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsProSACould"
        prompt_text = f"Could you provide a response to the following question:\n{{question}}\n{{choices}}\nAnswer:"
        return prompt_name, prompt_text

    def get_here_prompt_State_of_What_Art_paper(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsStateHere"
        prompt_text = f"Here are some multiple choice questions along with their answers about {{topic}}.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"
        return prompt_name, prompt_text

    def get_below_prompt_State_of_What_Art_paper(self) -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsStateBelow"
        prompt_text = f"Below are multiple-choice questions related to {{topic}}, each followed by their respective answers.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"
        return prompt_name, prompt_text

    def get_below_please_prompt_State_of_What_Art_paper(self)  -> PromptInstruction:
        prompt_name = "MultipleChoiceTemplatesInstructionsStateBelowPlease"
        prompt_text = f"Below are multiple-choice questions related to {{topic}}. Please provide the correct answer for each question.\n\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
        return prompt_name, prompt_text


if __name__ == "__main__":
    config = MMLUConfig({"enumerator": "numbers"})
    config_dict = config.to_dict()
    print(config_dict)
