from src.experiments.experiment_preparation.datasets_configurations.MMLUConfig import MMLUConfig


class AI2ARCChallengeConfig(MMLUConfig):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)


if __name__ == "__main__":
    config = AI2ARCChallengeConfig({"shuffle_choices": True})
    config_dict = config.to_dict()
    print(config_dict)
