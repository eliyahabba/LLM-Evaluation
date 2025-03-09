from dataclasses import dataclass

from src.experiments.experiment_preparation.InstructionPhrasings.BasicMCPrompts import BasicMCPrompts
from src.experiments.experiment_preparation.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class QuALITYPrompts(RacePrompts):
    pass