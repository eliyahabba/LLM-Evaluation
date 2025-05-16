#!/usr/bin/env python3
"""
Script to split the SimpleQA dataset into train and test sets.

This script:
1. Reads the existing SimpleQA CSV file
2. Randomly splits the data into train (80%) and test (20%) sets
3. Creates a new CSV file with an additional 'split' column
"""

import os
import csv
import random
import sys
from pathlib import Path

from src.data_augmentation.config.constants import (
    SIMPLE_QA_CSV_PATH, 
    DATA_DIR,
    GLOBAL_RANDOM_SEED
)

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


# Train/test split ratio
TRAIN_RATIO = 0.1

def split_simple_qa_csv(input_file=SIMPLE_QA_CSV_PATH, train_ratio=TRAIN_RATIO, seed=GLOBAL_RANDOM_SEED):
    """
    Split the SimpleQA dataset into train and test sets.
    
    Args:
        input_file (Path): Path to the SimpleQA CSV file
        train_ratio (float): Ratio of examples to put in the training set
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (output_file_path, num_train_examples, num_test_examples)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all examples from the CSV file
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)  # Get the header
        
        for row in csv_reader:
            examples.append(row)
    
    # Shuffle the examples
    random.shuffle(examples)
    
    # Split into train and test sets
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    # Add the split information to each example
    for example in train_examples:
        # Add 'train' as the split field
        example.append('train')
    
    for example in test_examples:
        # Add 'test' as the split field
        example.append('test')
    
    # Create output file path
    output_file = DATA_DIR / "simple_qa_with_split.csv"
    
    # Write the output CSV file with the split column
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        # Add 'split' to the header
        new_header = header + ['split']
        csv_writer.writerow(new_header)
        
        # Write all examples with their splits
        csv_writer.writerows(train_examples + test_examples)
    
    return output_file, len(train_examples), len(test_examples)

def main():
    """Main function to run the script."""
    # Initialize random seed
    random.seed(GLOBAL_RANDOM_SEED)
    
    print(f"Reading SimpleQA dataset from: {SIMPLE_QA_CSV_PATH}")
    print(f"Using random seed: {GLOBAL_RANDOM_SEED}")
    
    # Split the dataset
    output_file, num_train, num_test = split_simple_qa_csv()
    
    # Print statistics
    total = num_train + num_test
    print(f"\nSuccessfully split SimpleQA dataset:")
    print(f"Total examples: {total}")
    print(f"Training set: {num_train} examples ({num_train/total:.1%})")
    print(f"Test set: {num_test} examples ({num_test/total:.1%})")
    print(f"\nOutput file saved to: {output_file}")
    
    # Update the SimpleQA CSV path in constants.py (print instructions)
    print("\nTo use the new split dataset, update SIMPLE_QA_CSV in config/constants.py:")
    print(f'SIMPLE_QA_CSV = "simple_qa_with_split.csv"')

if __name__ == "__main__":
    main() 