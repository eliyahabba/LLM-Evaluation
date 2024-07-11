import sys
from pathlib import Path

import yaml

GLOBAL_CONFIG = Path(__file__).parent / 'global_config.yaml'
LOCAL_CONFIG = Path(__file__).parent / 'local_config.yaml'


class Config:
    def __init__(self):
        self.config_values = self.load_config()

    def load_config(self):
        config = {}
        try:
            with open(GLOBAL_CONFIG, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            print("Global config not found.", file=sys.stderr)

        try:
            with open(LOCAL_CONFIG, 'r') as file:
                local_config = yaml.safe_load(file)
                if local_config:
                    config.update(local_config)
        except FileNotFoundError:
            print("Local config not found.", file=sys.stderr)

        return config


if __name__ == "__main__":
    config = Config()
    # The key to lookup is passed as a command-line argument
    key = sys.argv[1]
    print(config.config_values.get(key, ''))
