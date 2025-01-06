from dataclasses import dataclass

from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.BasicMCPrompts import BasicMCPrompts


@dataclass(frozen=True)
class AI2ARCPrompts(BasicMCPrompts):
    pass
