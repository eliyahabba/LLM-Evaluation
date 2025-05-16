"""
Module for generating configuration files for the data augmentation system.
"""

import os
import json
import itertools
import random
from pathlib import Path
from src.data_augmentation.config.constants import (
    DATASETS, 
    TEXT_TRANSFORMATIONS, 
    FEW_SHOT,
    DATASET_INSTRUCTIONS,
    MAX_CONFIGS_PER_DATASET,
    EXAMPLES_PER_CONFIG,
    get_config_dir,
    qa_prompts,
    math_prompts,
    GLOBAL_RANDOM_SEED
)


class ConfigGenerator:
    """
    Generate configuration files for the data augmentation system.
    """
    
    def __init__(self, random_seed=GLOBAL_RANDOM_SEED):
        """
        Initialize the ConfigGenerator.
        
        Args:
            random_seed (int): Random seed for deterministic behavior
        """
        self.random_seed = random_seed
        # Initialize random seed
        random.seed(self.random_seed)
        
    def generate_dataset_configs(self, dataset):
        """
        Generate all possible configurations for a specific dataset.
        
        Args:
            dataset (str): The dataset name
            
        Returns:
            list: List of configuration dictionaries for the dataset
        """
        configs = []
        
        # Get templates for this dataset based on dataset type
        if dataset == "simple_qa":
            dataset_templates = qa_prompts
        elif dataset == "gsm8k" or dataset == "math":
            dataset_templates = math_prompts
        else:
            # Fallback to the old format
            dataset_templates = DATASET_INSTRUCTIONS.get(dataset, [])
        
        # Get all combinations of parameters
        config_id = 1
        
        # Generate all possible combinations
        all_combinations = []
        
        # With the new 3-part format, our templates are dictionaries
        for template_name, template in dataset_templates.items():
            for transformation in TEXT_TRANSFORMATIONS:
                for count in FEW_SHOT["count_range"]:
                    for seed in FEW_SHOT["seed_range"]:
                        # Store the combination
                        combination = (template_name, template, transformation, count, seed)
                        all_combinations.append(combination)
        
        # If we have more combinations than the maximum allowed, randomly sample
        if len(all_combinations) > MAX_CONFIGS_PER_DATASET:
            # Use our initialized seed for reproducibility
            random.seed(self.random_seed)
            all_combinations = random.sample(all_combinations, MAX_CONFIGS_PER_DATASET)
        
        # Create configurations from the selected combinations
        for template_name, template, transformation, count, seed in all_combinations:
            # Generate a different examples_seed for each configuration
            # This ensures different examples are selected for each config
            # But we want it to be deterministic based on our global seed
            examples_seed = random.randint(1000, 9999)
            
            # Create a configuration with the 3-part structure
            config = {
                "id": f"{config_id:03d}",
                "template_name": template_name,
                "instruction": template["instruction"],
                "input_format": template["input_format"],
                "target_prefix": template["target_prefix"],
                "text_transformations": {
                    "technique": transformation["name"],
                    "parameters": transformation["params"]
                },
                "few_shot": {
                    "count": count,
                    "random_seed": seed
                },
                "examples_seed": examples_seed,  # Add seed for example selection
                "num_examples": EXAMPLES_PER_CONFIG,  # Number of examples to select
                "dataset": dataset
            }
            
            configs.append(config)
            config_id += 1
        
        return configs
    
    def generate_all_configs(self):
        """
        Generate configurations for all datasets.
        
        Returns:
            dict: Dictionary with dataset names as keys and lists of configs as values
        """
        all_configs = {}
        
        for dataset in DATASETS.keys():
            dataset_configs = self.generate_dataset_configs(dataset)
            all_configs[dataset] = dataset_configs
            print(f"Generated {len(dataset_configs)} configurations for {dataset}")
        
        return all_configs
    
    def generate_config_filename(self, config):
        """
        Generate a descriptive filename for a configuration file.
        
        Args:
            config (dict): The configuration dictionary
            
        Returns:
            str: A descriptive filename
        """
        template_name = config["template_name"]
        transformation = config["text_transformations"]["technique"]
        few_shot_count = config["few_shot"]["count"]
        few_shot_seed = config["few_shot"]["random_seed"]
        examples_seed = config["examples_seed"]
        
        # Create a descriptive filename that includes the examples seed
        filename = f"{template_name}_{transformation}_fewshot{few_shot_count}_seed{few_shot_seed}_exseed{examples_seed}.json"
        
        return filename
    
    def save_configs(self, all_configs):
        """
        Save configuration files for all datasets.
        
        Args:
            all_configs (dict): Dictionary with dataset names as keys and lists of configs as values
            
        Returns:
            dict: Dictionary with dataset names as keys and lists of saved file paths as values
        """
        saved_files = {}
        
        for dataset, configs in all_configs.items():
            # Get config directory for this dataset
            config_dir = get_config_dir(dataset)
            config_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_files = []
            
            for config in configs:
                # Generate a descriptive filename
                descriptive_filename = self.generate_config_filename(config)
                
                # Store the original ID for reference inside the config
                config["id_original"] = config["id"]
                
                # Add a field with the descriptive name (without extension) for reference
                config["descriptive_name"] = descriptive_filename.replace('.json', '')
                
                # Use only the descriptive filename to match data file naming
                config_file = config_dir / descriptive_filename
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                    
                dataset_files.append(str(config_file))
            
            saved_files[dataset] = dataset_files
            print(f"Saved {len(dataset_files)} configuration files for {dataset}")
            
        return saved_files 