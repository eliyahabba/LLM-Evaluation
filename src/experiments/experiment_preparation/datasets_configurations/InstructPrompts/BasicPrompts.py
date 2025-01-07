from dataclasses import dataclass, fields

from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.Instruction import \
    Instruction


@dataclass(frozen=True)
class BasicPrompts:
    def get_all_prompts(self):
        return [
            getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Instruction)
        ]
