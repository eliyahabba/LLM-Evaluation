from src.experiments.experiment_preparation.datasets_configurations import BoolQConfig
from src.experiments.experiment_preparation.datasets_configurations.AI2ARCChallengeConfig import AI2ARCChallengeConfig
from src.experiments.experiment_preparation.datasets_configurations.BoolQConfig import BoolQConfig
from src.experiments.experiment_preparation.datasets_configurations.HellaSwagConfig import HellaSwagConfig
from src.experiments.experiment_preparation.datasets_configurations.MMLUConfig import MMLUConfig
from src.utils.MMLUData import MMLUData


class DatasetConfigFactory:
    @staticmethod
    def get_create_mmlu_config() -> dict:
        mmlu_dataset_classes = {}
        MMLUData.initialize()
        for mmlu_dataset in MMLUData.get_mmlu_datasets():
            mmlu_dataset_classes[f"{mmlu_dataset}"] = MMLUConfig
        return mmlu_dataset_classes

    @staticmethod
    def get_create_mmlu_pro_config() -> dict:
        mmlu_dataset_classes = {}
        MMLUData.initialize()
        for mmlu_dataset in MMLUData.get_mmlu_pro_datasets():
            mmlu_dataset_classes[f"{mmlu_dataset}"] = MMLUConfig
        return mmlu_dataset_classes

    @staticmethod
    def get_dataset(dataset_name):
        dataset_classes = {
            'ai2_arc.arc_challenge': AI2ARCChallengeConfig,
            'boolq.multiple_choice': BoolQConfig,
            'hellaswag': HellaSwagConfig,
            # Add other datasets here
        }
        mmlu_dataset_classes = DatasetConfigFactory.get_create_mmlu_config()
        mmlu_pro_dataset_classes = DatasetConfigFactory.get_create_mmlu_pro_config()
        dataset_classes.update(mmlu_dataset_classes)
        dataset_classes.update(mmlu_pro_dataset_classes)
        return {dataset_name: dataset_classes[dataset_name]}

    @staticmethod
    def get_all_datasets():
        mmlu_dataset_classes = DatasetConfigFactory.get_create_mmlu_config()
        mmlu_pro_dataset_classes = DatasetConfigFactory.get_create_mmlu_pro_config()
        dataset_classes = {
            'ai2_arc.arc_challenge': AI2ARCChallengeConfig,
            'boolq.multiple_choice': BoolQConfig,
            'hellaswag': HellaSwagConfig,
            # Add other datasets here
        }
        dataset_classes.update(mmlu_dataset_classes)
        dataset_classes.update(mmlu_pro_dataset_classes)
        return dataset_classes
