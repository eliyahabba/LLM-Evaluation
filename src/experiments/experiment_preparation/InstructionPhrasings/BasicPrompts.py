from dataclasses import dataclass, fields

from src.experiments.experiment_preparation.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class BasicPrompts:
    def get_instruction_phrasings(self):
        return [
            getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Instruction)
        ]
