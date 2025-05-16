"""
Module for processing datasets according to configurations.
"""

import os
import json
import random
from pathlib import Path
from src.data_augmentation.utils.text_surface_augmenter import TextSurfaceAugmenter
from src.data_augmentation.config.constants import (
    PROCESSED_DATA_DIR,
    get_data_dir,
    EXAMPLES_PER_CONFIG,
    GLOBAL_RANDOM_SEED,
    MAX_ANSWER_TOKENS
)


class DataProcessor:
    """
    Process datasets according to configuration parameters.
    """
    
    def __init__(self, max_examples=None, random_seed=GLOBAL_RANDOM_SEED):
        """
        Initialize the DataProcessor.
        
        Args:
            max_examples (int, optional): Maximum number of examples to process
            random_seed (int): Random seed for deterministic behavior
        """
        self.max_examples = max_examples
        self.random_seed = random_seed
        self.augmenter = TextSurfaceAugmenter(n_augments=1, random_seed=self.random_seed)
        # Initialize random generator
        random.seed(self.random_seed)
        
    def sample_examples(self, examples, num_examples, seed, dataset_type=None):
        """
        Sample a random subset of examples based on the provided seed.
        
        Args:
            examples (list): All available examples
            num_examples (int): Number of examples to sample
            seed (int): Random seed for reproducibility
            dataset_type (str, optional): Type of dataset for special handling
            
        Returns:
            list: Sampled examples
        """
        # Set seed for reproducibility
        random.seed(seed)
        
        # Ensure num_examples is an integer
        if isinstance(num_examples, dict):
            # If num_examples is somehow passed as a dictionary (EXAMPLES_PER_CONFIG),
            # try to get the value for this dataset_type
            num_examples = num_examples.get(dataset_type, 1000)
        
        # Apply token count filtering if configured for this dataset
        max_tokens = MAX_ANSWER_TOKENS.get(dataset_type, None)
        filtered_examples = examples
        
        if max_tokens is not None:
            # Filter examples where answer_tokens is not greater than max_tokens
            filtered_examples = [
                ex for ex in examples 
                if "answer_tokens" not in ex or 
                   ex["answer_tokens"] is None or 
                   ex["answer_tokens"] <= max_tokens
            ]
            print(f"Filtered examples for {dataset_type}: {len(examples)} -> {len(filtered_examples)} " 
                  f"(max tokens: {max_tokens})")
            
            # If all examples were filtered out, use the original list
            if not filtered_examples:
                print(f"Warning: All examples were filtered out for {dataset_type}. Using original examples.")
                filtered_examples = examples
        
        # For GSM8K or SimpleQA, use only test examples for evaluation
        if dataset_type in ["gsm8k", "simple_qa"]:
            test_examples = [ex for ex in filtered_examples if ex.get("split") == "test"]
            if test_examples:
                # Make a copy to avoid affecting the original list
                examples_copy = test_examples.copy()
                # Shuffle and select examples
                random.shuffle(examples_copy)
                selected = examples_copy[:min(num_examples, len(examples_copy))]
                return selected
        
        # Default behavior for other datasets or if no test examples found
        # Make a copy to avoid affecting the original list
        examples_copy = filtered_examples.copy()
        
        # Shuffle and select examples
        random.shuffle(examples_copy)
        selected = examples_copy[:min(num_examples, len(examples_copy))]
        
        return selected
    
    def select_few_shot_examples(self, examples, count, seed, dataset_type=None):
        """
        Select a subset of examples to use as few-shot examples.
        
        Args:
            examples (list): All available examples
            count (int): Number of examples to select
            seed (int): Random seed for reproducibility
            dataset_type (str, optional): Type of dataset for special handling
            
        Returns:
            list: Selected few-shot examples
        """
        if count == 0:
            return []
            
        # Set seed for reproducibility
        random.seed(seed)
        
        # For GSM8K or SimpleQA, use only training examples for few-shot
        if dataset_type in ["gsm8k", "simple_qa"]:
            training_examples = [ex for ex in examples if ex.get("split") == "train"]
            if training_examples:
                # Make a copy to avoid affecting the original list
                examples_copy = training_examples.copy()
                # Shuffle and select the first 'count' examples
                random.shuffle(examples_copy)
                selected = examples_copy[:count]
                return selected
        
        # Default behavior for other datasets or if no training examples found
        # Make a copy to avoid affecting the original list
        examples_copy = examples.copy()
        
        # Shuffle and select the first 'count' examples
        random.shuffle(examples_copy)
        selected = examples_copy[:count]
        
        return selected
    
    def apply_transformation(self, text, technique, parameters):
        """
        Apply a specific transformation technique to the text.
        
        Args:
            text (str): Original text to transform
            technique (str): Transformation technique to apply
            parameters (dict): Parameters for the transformation
            
        Returns:
            str: Transformed text
        """
        # Handle no_transform case specifically
        if technique == "no_transform" or parameters.get("disable_transformations", False):
            return text
            
        # Create an augmenter instance
        augmenter = TextSurfaceAugmenter(n_augments=1)
        
        # Apply the specified technique
        if technique in ["spacing", "spacing_light", "spacing_extreme"]:
            probability = parameters.get("probability", 0.8)
            min_spaces = parameters.get("min_spaces", 1)
            max_spaces = parameters.get("max_spaces", 3)
            word_probability = parameters.get("word_probability", 0.5)
            
            transformed = augmenter.add_white_spaces(
                text, 
                probability=probability, 
                min_spaces=min_spaces, 
                max_spaces=max_spaces, 
                word_probability=word_probability
            )
        
        elif technique == "no_spacing":
            # Use preserve_original=True to keep the original spacing
            preserve_original = parameters.get("preserve_original", True)
            transformed = augmenter.add_white_spaces(text, preserve_original=preserve_original)
            
        elif technique == "case_change":
            probability = parameters.get("probability", 0.1)
            transformed = augmenter.change_char_case(text, prob=probability, max_outputs=1)[0]
            
        elif technique == "typo":
            probability = parameters.get("probability", 0.05)
            transformed = augmenter.butter_finger(text, prob=probability, max_outputs=1)[0]
            
        elif technique == "character_swap":
            probability = parameters.get("probability", 0.08)
            transformed = augmenter.swap_characters(text, prob=probability, max_outputs=1)[0]
            
        elif technique == "punctuation_swap":
            probability = parameters.get("probability", 0.1)
            transformed = augmenter.switch_punctuation(text, prob=probability, max_outputs=1)[0]
            
        else:
            # If technique is not recognized, return the original text
            transformed = text
            
        return transformed
    
    def transform_example(self, example, technique, parameters, dataset_type):
        """
        Apply transformations to an example.
        
        Args:
            example (dict): The example to transform
            technique (str): The transformation technique to apply
            parameters (dict): Parameters for the transformation
            dataset_type (str): Type of dataset for special handling
            
        Returns:
            dict: Transformed example with all required fields
        """
        # Start with a copy of the original example to preserve all fields
        transformed = example.copy() if isinstance(example, dict) else {"question": example}
        
        # Make sure we have the required fields
        if "id" not in transformed and "key" in example:
            transformed["id"] = example["key"]
            
        # Extract the question based on dataset type
        if dataset_type == "simple_qa":
            question_field = "question"
        else:  # GSM8K or MATH
            question_field = "question"
        
        # Apply transformation to the question
        original_text = transformed.get(question_field, "")
        if not original_text and "question" in example:
            original_text = example["question"]
            
        # Save the original question for reference
        if "original_question" not in transformed:
            transformed["original_question"] = original_text
        
        # Skip transformation if "no_transform" technique is specified
        if technique == "no_transform" or parameters.get("disable_transformations", False):
            # Keep the original text without transformation
            transformed_text = original_text
        else:
            # Apply the transformation
            transformed_text = self.apply_transformation(original_text, technique, parameters)
            
        transformed[question_field] = transformed_text
        
        # Ensure there's an answer if available in the original
        if "answer" in example and "answer" not in transformed:
            transformed["answer"] = example["answer"]
            
        return transformed
    
    def format_instruction_with_problem(self, template, example, dataset_type):
        """
        Format an instruction template with problem content.
        
        Args:
            template (dict): Template containing instruction, input_format, and target_prefix
            example (dict): Example to format with the template
            dataset_type (str): Type of dataset
            
        Returns:
            str: Formatted instruction with problem content
        """
        # Extract relevant fields based on dataset type
        if dataset_type == "simple_qa":
            problem_text = example.get("question", "")
            answer_text = example.get("answer", "")
            placeholder = "{question}"
        elif dataset_type == "gsm8k" or dataset_type == "math":
            problem_text = example.get("question", "")
            answer_text = example.get("answer", "")
            placeholder = "{problem}"
        else:
            # Default behavior
            problem_text = example.get("question", "")
            answer_text = example.get("answer", "")
            placeholder = "{question}"
        
        # Extract the three components from the template
        instruction = template.get("instruction", "")
        input_format = template.get("input_format", "")
        target_prefix = template.get("target_prefix", "")
        
        # Format the input by replacing the placeholder with the problem text
        formatted_input = input_format.replace(placeholder, problem_text)
        
        # Combine the components to create the final formatted instruction
        if instruction:
            formatted_instruction = f"{instruction}\n{formatted_input}{target_prefix} {answer_text}"
        else:
            formatted_instruction = f"{formatted_input}{target_prefix} {answer_text}"
        
        return formatted_instruction
    
    def process_dataset_with_config(self, examples, config):
        """
        Process a dataset with a specific configuration.
        
        Args:
            examples (list): Examples to process
            config (dict): Configuration parameters
            
        Returns:
            list: Processed examples according to the format required
        """
        # Get parameters from config
        dataset_type = config.get("dataset")
        instruction = config["instruction"]
        input_format = config["input_format"]
        target_prefix = config["target_prefix"]
        technique = config["text_transformations"]["technique"]
        parameters = config["text_transformations"]["parameters"]
        few_shot_count = config["few_shot"]["count"]
        random_seed = config["few_shot"]["random_seed"]
        
        # Sample examples using the config's random seed
        # Get the number of examples based on dataset type (use dataset's value from EXAMPLES_PER_CONFIG)
        default_num_examples = EXAMPLES_PER_CONFIG.get(dataset_type, 1000)  # Default to 1000 if dataset not in dictionary
        num_examples = config.get("num_examples", default_num_examples)
        # Use the configuration's random seed for sampling
        examples_seed = config.get("examples_seed", random_seed)  # Fallback to few_shot seed if not specified
        sampled_examples = self.sample_examples(examples, num_examples, examples_seed, dataset_type)
        
        # Select few-shot examples
        few_shot_examples = self.select_few_shot_examples(
            examples, few_shot_count, random_seed, dataset_type
        )
        
        # Transform few-shot examples
        transformed_few_shot = []
        for fs_example in few_shot_examples:
            transformed_fs = self.transform_example(fs_example, technique, parameters, dataset_type)
            
            # Format the few-shot example with the template structure from the config
            if dataset_type == "simple_qa":
                formatted_question = input_format.replace("{question}", transformed_fs["question"])
            else:
                formatted_question = input_format.replace("{problem}", transformed_fs["question"])
                
            # Format the few-shot example to contain the formatted question and answer
            fs_formatted = {
                "question": formatted_question,
                "answer": f"{transformed_fs.get('answer', '')}"
            }
            
            transformed_few_shot.append(fs_formatted)
        
        # Process each example from the sampled set
        processed_examples = []
        for example in sampled_examples:
            # Apply transformations to the question
            transformed_example = self.transform_example(example, technique, parameters, dataset_type)
            
            # Format the processed example using the config's template values
            processed = {
                "id": f"{dataset_type}_{example.get('id', 'unknown')}",
                "instruction": instruction,
                "question": transformed_example["question"],
                "few_shot_examples": transformed_few_shot,
                "original_question": example["question"],
                "input_format": input_format,
                "target_prefix": target_prefix
            }
            
            processed_examples.append(processed)
        
        return processed_examples
    
    def save_processed_data(self, processed_examples, config, output_file=None):
        """
        Save processed data to a JSON file in the chat format.
        
        Args:
            processed_examples (list): List of processed examples
            config (dict): Configuration used to process the examples
            output_file (str, optional): Path to save the file
            
        Returns:
            str: Path to the saved file
        """
        dataset = config["dataset"]
        config_id = config["id"]
        
        if output_file is None:
            # Get data directory for this dataset
            data_dir = get_data_dir(dataset)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if the config has a descriptive name
            if "descriptive_name" in config:
                # Use the same exact filename as the config file
                output_file = data_dir / f"{config['descriptive_name']}.json"
            else:
                # Fallback to using just the ID (should not occur with updated system)
                output_file = data_dir / f"data_{config_id}.json"
        
        # Ensure output_file is a Path object
        output_file = Path(output_file)
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Transform the processed examples into the new chat format
        chat_format_data = self._create_chat_format(processed_examples, config)
            
        # Save the processed examples in the new format
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chat_format_data, f, indent=2)
            
        return str(output_file)
    
    def _create_chat_format(self, processed_examples, config):
        """
        Transform processed examples into the new chat format.
        
        Args:
            processed_examples (list): List of processed examples
            config (dict): Configuration used to process the examples
            
        Returns:
            dict: Data in the new chat format
        """
        # Get the descriptive name to use as the main key
        name = config.get("descriptive_name", config["id"])
        
        # Create the structure with the name as the main key
        chat_format = {
            name: {
                "source": []
            }
        }
        
        # Process each example into a conversation
        for example in processed_examples:
            # Get the components
            instruction = example["instruction"]
            input_format = example["input_format"].replace("{problem}", example["question"]) if "{problem}" in example["input_format"] else example["input_format"].replace("{question}", example["question"])
            target_prefix = example["target_prefix"]
            few_shot_examples = example.get("few_shot_examples", [])
            
            # Create a conversation (array of messages)
            conversation = []
            
            # Add the few-shot examples first
            for i, fs_example in enumerate(few_shot_examples):
                fs_question = fs_example["question"]
                fs_answer = fs_example["answer"]
                
                # Add the user message (question)
                if i == 0:
                    # First few-shot includes the instruction
                    user_content = f"{instruction}{fs_question}{target_prefix}"
                else:
                    # Subsequent few-shots don't repeat the instruction
                    user_content = f"{fs_question}{target_prefix}"
                
                conversation.append({
                    "role": "user",
                    "content": user_content
                })
                
                # Add the assistant message (answer)
                conversation.append({
                    "role": "assistant",
                    "content": fs_answer
                })
            
            # Add the actual question at the end (no assistant answer since this is what we're testing)
            if few_shot_examples:
                # If we had few-shot examples, don't repeat the instruction
                user_content = f"{input_format}{target_prefix}"
            else:
                # If no few-shot, include the instruction
                user_content = f"{instruction}{input_format}{target_prefix}"
                
            conversation.append({
                "role": "user",
                "content": user_content
            })
            
            # Add the conversation to the source array
            chat_format[name]["source"].append(conversation)
            
            # Add the original ID to the data structure
            if "id" in example:
                if "ids" not in chat_format[name]:
                    chat_format[name]["ids"] = []
                chat_format[name]["ids"].append(example["id"])
        
        return chat_format
    
    def process_configs(self, all_examples, all_configs):
        """
        Process examples using multiple configurations.
        
        Args:
            all_examples (dict): Dictionary mapping dataset names to lists of examples
            all_configs (dict): Dictionary mapping dataset names to lists of configurations
            
        Returns:
            dict: Mapping of dataset names to lists of processed data file paths
        """
        results = {}
        
        for dataset, configs in all_configs.items():
            dataset_results = []
            
            # Get examples for this dataset
            examples = all_examples.get(dataset, [])
            if not examples:
                print(f"Warning: No examples found for dataset '{dataset}'. Skipping.")
                continue
                
            print(f"Processing {len(configs)} configurations for dataset '{dataset}'...")
            
            for config in configs:
                # Process examples with this configuration
                processed_examples = self.process_dataset_with_config(examples, config)
                
                # Save processed data
                output_file = self.save_processed_data(processed_examples, config)
                dataset_results.append(output_file)
                
                print(f"  - Processed {len(processed_examples)} examples for config {config['id']}")
                
            results[dataset] = dataset_results
            print(f"Processed {len(dataset_results)} configurations for dataset '{dataset}'")
            
        return results 