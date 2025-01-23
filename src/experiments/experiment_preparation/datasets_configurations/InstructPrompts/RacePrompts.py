from dataclasses import dataclass

from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.BasicPrompts import BasicPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.Instruction import \
    Instruction


@dataclass(frozen=True)
class RacePrompts(BasicPrompts):
    race_instructions_basic: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"Context: {{context}}\nQuestion: {{question}}\nChoices:\n{{choices}}\nAnswer:"
    )

    race_instructions_basic_without_Context: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"{{context}}\nQuestion: {{question}}\nChoices:\n{{choices}}\nAnswer:"
    )

    mmlu_instructions = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"The following are multiple choice questions (with answers).\n\nContext: {{context}}\n{{question}}\n{{choices}}\nAnswer:"
    )

    mmlu_instructions_helm = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"The following are multiple choice questions (with answers).\n\nContext: {{context}}\nQuestion: {{question}}\n{{choices}}\nAnswer:"
    )

    mmlu_instructions_helm_with_Choices = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"The following are multiple choice questions (with answers).\n\nContext: {{context}}\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )

    please_simple_prompt_ProSA_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsProSASimple",
        text=f"Please answer the following question based on the article:\nContext: {{context}}\n{{question}}\n{{choices}}\nAnswer:"
    )

    could_you_prompt_ProSA_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsProSACould",
        text=f"Could you provide a response to the following question based on the article:\nContext: {{context}}\n{{question}}\n{{choices}}\nAnswer:"
    )

    Answer_prompt_State_of_What_Art_paper = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=(
            f"Answer the multiple choice Question from one of the Choices (choose from numerals) based on the context.\n"
            f"Context: {{context}}\nQuestion: {{question}}\nChoices:\n{{choices}}\nAnswer:")
    )

    Answer_prompt_State_of_What_Art_paper2 = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"Based on the provided article, please respond to the following question:\nContext: {{context}}\nQuestion: {{question}}\nOptions: {{choices}}\nAnswer:"
    )

    Answer_prompt_State_of_What_Art_paper3 = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"Select the correct answer for the following multiple-choice question based on the context. Choose from the numbered options provided.\nContext: {{context}}\nQuestion: {{question}}\nOptions: {{choices}}\nAnswer:"""
    )

    Answer_prompt_State_of_What_Art_paper4 = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"Read the following context and answer the question:\n{{context}}\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )

    Answer_prompt_State_of_What_Art_paper5 = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"Read the following context and answer the question:\n{{context}}\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )

    Answer_prompt_State_of_What_Art_paper6 = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"Answer the multiple-choice question below about the context:\n\nContext: {{context}}\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:”
    )