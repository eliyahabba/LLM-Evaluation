from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.AI2ARCPrompts import AI2ARCPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.BasicMCPrompts import BasicMCPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.HellaSwagPrompts import \
    HellaSwagPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.MMLUPrompts import MMLUPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.OpenBookQAPrompts import \
    OpenBookQAPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.SocialQaPrompts import \
    SocialQaPrompts


class DatasetConfigFactory:
    @staticmethod
    def get_all_instruct_prompts():
        dataset_instruct_prompts = {
            'AI2-ARC': AI2ARCPrompts(),
            'HellaSwag': HellaSwagPrompts(),
            'MMLU': MMLUPrompts(),
            'OpenBookQA': OpenBookQAPrompts(),
            'Social-IQa': SocialQaPrompts()
            # Add other datasets here
        }
        return dataset_instruct_prompts

    @staticmethod
    def get_instruct_prompts(dataset_name):
        dataset_instruct_prompts = {
            'AI2-ARC': AI2ARCPrompts(),
            'HellaSwag': HellaSwagPrompts(),
            'MMLU': MMLUPrompts(),
            'OpenBookQA': OpenBookQAPrompts(),
            'Social-IQa': SocialQaPrompts()
            # Add other datasets here
        }
        return BasicMCPrompts() if dataset_name not in dataset_instruct_prompts else \
            dataset_instruct_prompts[dataset_name]
