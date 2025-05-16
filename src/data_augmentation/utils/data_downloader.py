"""
Module for downloading and preparing datasets from Hugging Face.
"""

import os
import json
import csv
import ast
import random
from datasets import load_dataset
from src.data_augmentation.config.constants import (
    SIMPLE_QA_CSV_PATH, 
    GSM8K_CSV_PATH,
    GLOBAL_RANDOM_SEED
)


class DataDownloader:
    """
    A class for downloading and preparing datasets from Hugging Face.
    """

    def __init__(self, cache_dir=None, random_seed=GLOBAL_RANDOM_SEED):
        """
        Initialize the DataDownloader.
        
        Args:
            cache_dir (str, optional): Directory to cache the downloaded datasets
            random_seed (int): Random seed for deterministic behavior
        """
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        # Initialize random seed
        random.seed(self.random_seed)
        
    def download_dataset(self, dataset_name, dataset_config=None, split="train"):
        """
        Download a dataset from Hugging Face.
        
        Args:
            dataset_name (str): Name of the dataset on Hugging Face
            dataset_config (str, optional): Specific configuration of the dataset
            split (str, optional): Split of the dataset to download
            
        Returns:
            Dataset: The downloaded dataset
        """
        try:
            dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                split=split,
                cache_dir=self.cache_dir
            )
            return dataset
        except Exception as e:
            print(f"Error downloading dataset {dataset_name}: {e}")
            return None
    
    def load_simple_qa_from_csv(self, file_path=SIMPLE_QA_CSV_PATH):
        """
        Load SimpleQA dataset from a CSV file.
        
        Args:
            file_path (str, optional): Path to the CSV file. Defaults to SIMPLE_QA_CSV_PATH.
            
        Returns:
            list: A list of examples
        """
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                # Skip header row
                header = next(csv_reader)
                
                # Check if answer_tokens column exists
                answer_tokens_idx = -1
                for i, col in enumerate(header):
                    if col.lower() == 'answer_tokens':
                        answer_tokens_idx = i
                        break
                
                for i, row in enumerate(csv_reader):
                    if len(row) >= 3:  # Ensure row has metadata, problem, and answer
                        # Try to parse metadata as dictionary
                        try:
                            metadata = ast.literal_eval(row[0])
                        except (SyntaxError, ValueError):
                            metadata = {"raw_metadata": row[0]}
                            
                        example = {
                            "id": str(i),
                            "question": row[1],
                            "answer": row[2],
                            "metadata": metadata
                        }
                        
                        # Add answer_tokens if the column exists
                        if answer_tokens_idx >= 0 and len(row) > answer_tokens_idx:
                            try:
                                answer_tokens = int(row[answer_tokens_idx])
                                example["answer_tokens"] = answer_tokens
                            except (ValueError, TypeError):
                                # If conversion fails, store as None
                                example["answer_tokens"] = None
                                
                        examples.append(example)
                    else:
                        print(f"Warning: Row {i+1} has fewer than 3 columns. Skipping.")
        except Exception as e:
            print(f"Error loading SimpleQA from CSV: {e}")
            
        return examples
    
    def load_gsm8k_from_csv(self, file_path=GSM8K_CSV_PATH):
        """
        Load GSM8K dataset from a CSV file.
        
        Args:
            file_path (str, optional): Path to the CSV file. Defaults to GSM8K_CSV_PATH.
            
        Returns:
            list: A list of examples
        """
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                # Skip header row
                header = next(csv_reader)
                
                # Check if answer_tokens column exists
                answer_tokens_idx = -1
                for i, col in enumerate(header):
                    if col.lower() == 'answer_tokens':
                        answer_tokens_idx = i
                        break
                
                for i, row in enumerate(csv_reader):
                    if len(row) >= 3:  # Ensure row has id, question, and answer
                        # Try to parse metadata as dictionary
                        metadata = {}
                        if len(row) >= 5:  # If metadata column exists
                            try:
                                metadata = json.loads(row[4])
                            except (json.JSONDecodeError, IndexError):
                                metadata = {"raw_metadata": row[4] if len(row) > 4 else ""}
                        
                        example = {
                            "id": row[0],
                            "question": row[1],
                            "answer": row[2],
                            "split": row[3] if len(row) > 3 else "unknown",
                            "metadata": metadata
                        }
                        
                        # Add answer_tokens if the column exists
                        if answer_tokens_idx >= 0 and len(row) > answer_tokens_idx:
                            try:
                                answer_tokens = int(row[answer_tokens_idx])
                                example["answer_tokens"] = answer_tokens
                            except (ValueError, TypeError):
                                # If conversion fails, store as None
                                example["answer_tokens"] = None
                        
                        examples.append(example)
                    else:
                        print(f"Warning: Row {i+1} has fewer than 3 columns. Skipping.")
        except Exception as e:
            print(f"Error loading GSM8K from CSV: {e}")
            
        return examples
    
    def prepare_simple_qa(self, dataset):
        """
        Prepare the SimpleQA dataset for use in the augmentation system.
        
        Args:
            dataset (Dataset): The loaded SimpleQA dataset
            
        Returns:
            list: A list of formatted examples
        """
        examples = []
        
        for i, item in enumerate(dataset):
            example = {
                "id": str(i) if "id" not in item else item["id"],
                "question": item["question"],
                "answer": item["answer"] if "answer" in item else item["answers"][0] if "answers" in item else ""
            }
            examples.append(example)
            
        return examples
    
    def prepare_gsm8k(self, dataset):
        """
        Prepare the GSM8K dataset for use in the augmentation system.
        
        Args:
            dataset (Dataset): The loaded GSM8K dataset
            
        Returns:
            list: A list of formatted examples
        """
        examples = []
        
        for item in dataset:
            example = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"]
            }
            examples.append(example)
            
        return examples
    
    def prepare_math(self, dataset):
        """
        Prepare the MATH dataset for use in the augmentation system.
        
        Args:
            dataset (Dataset): The loaded MATH dataset
            
        Returns:
            list: A list of formatted examples
        """
        examples = []
        
        for i, item in enumerate(dataset):
            example = {
                "id": str(i) if "id" not in item else item["id"],
                "question": item["problem"],
                "answer": item["solution"]
            }
            examples.append(example)
            
        return examples
    
    def save_examples(self, examples, output_file):
        """
        Save examples to a JSON file.
        
        Args:
            examples (list): The list of examples to save
            output_file (str): Path to the output file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2)
    
    def download_and_prepare(self, dataset_type, output_file=None):
        """
        Download and prepare a specific dataset.
        
        Args:
            dataset_type (str): Type of dataset ('simple_qa', 'gsm8k', or 'math')
            output_file (str, optional): Path to save the prepared examples
            
        Returns:
            list: The prepared examples
        """
        if dataset_type == "simple_qa":
            # For SimpleQA, load from CSV file
            examples = self.load_simple_qa_from_csv(SIMPLE_QA_CSV_PATH)
        elif dataset_type == "gsm8k":
            # For GSM8K, load from CSV file
            examples = self.load_gsm8k_from_csv(GSM8K_CSV_PATH)
        elif dataset_type == "math":
            dataset = self.download_dataset("competition_math")
            examples = self.prepare_math(dataset)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        if output_file:
            self.save_examples(examples, output_file)
            
        return examples 