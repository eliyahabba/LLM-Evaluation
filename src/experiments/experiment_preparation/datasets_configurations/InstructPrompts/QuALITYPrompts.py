from dataclasses import dataclass

from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.BasicPrompts import BasicPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.Instruction import \
    Instruction
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.RacePrompts import RacePrompts


@dataclass(frozen=True)
class QuALITYPrompts(RacePrompts):
    pass