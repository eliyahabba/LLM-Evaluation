from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.AI2ARCPrompts import AI2ARCPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.HellaSwagPrompts import \
    HellaSwagPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.MMLUPrompts import MMLUPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.OpenBookQAPrompts import \
    OpenBookQAPrompts
from src.experiments.experiment_preparation.datasets_configurations.InstructPrompts.SocialQaPrompts import \
    SocialQaPrompts


class DatasetConfigFactory:
    # @staticmethod
    # def get_dataset(dataset_name):
    #     dataset_classes = {
    #         'ai2_arc': AI2ARCChallengeConfig,
    #         'hellaswag': HellaSwagConfig,
    #         'mmlu': MMLUConfig,
    #         'open_book_qa': OpenBookQAConfig,
    #         'social_qa': SocialQaConfig
    #         # Add other datasets here
    #     }
    #     return {dataset_name: dataset_classes[dataset_name]}
    #
    @staticmethod
    def get_all_instruct_prompts():
        dataset_instruct_prompts = {
            'ai2_arc': AI2ARCPrompts(),
            'hellaswag': HellaSwagPrompts(),
            'mmlu': MMLUPrompts(),
            'open_book_qa': OpenBookQAPrompts(),
            'social_qa': SocialQaPrompts()
            # Add other datasets here
        }
        return dataset_instruct_prompts

    @staticmethod
    def get_instruct_prompts(dataset_name):
        dataset_instruct_prompts = {
            'ai2_arc': AI2ARCPrompts(),
            'hellaswag': HellaSwagPrompts(),
            'mmlu': MMLUPrompts(),
            'open_book_qa': OpenBookQAPrompts(),
            'social_qa': SocialQaPrompts()
            # Add other datasets here
        }
        return dataset_instruct_prompts[dataset_name]
