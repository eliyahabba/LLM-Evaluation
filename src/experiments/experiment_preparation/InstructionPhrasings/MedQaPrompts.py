from dataclasses import dataclass

from src.experiments.experiment_preparation.InstructionPhrasings.BasicMCPrompts import BasicMCPrompts
from src.experiments.experiment_preparation.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class OpenBookQAPrompts(BasicMCPrompts):
    instructions_with_topic_helm: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithTopicHelm",
        text=f"The following are multiple choice questions (with answers) about medicine.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"
    )