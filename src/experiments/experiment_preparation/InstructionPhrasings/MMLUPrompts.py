from dataclasses import dataclass

from src.experiments.experiment_preparation.InstructionPhrasings.BasicMCPrompts import BasicMCPrompts


@dataclass(frozen=True)
class MMLUPrompts(BasicMCPrompts):
    pass
