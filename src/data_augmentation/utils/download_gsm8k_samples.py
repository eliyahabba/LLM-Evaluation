"""
Script to download samples from GSM8K dataset and save to CSV.
"""

import os
import csv
import json
from datasets import load_dataset
from pathlib import Path
from src.data_augmentation.config.constants import GSM8K_CSV_PATH

def download_gsm8k_samples(num_train=100, num_test=2000, output_file=None):
    """
    Download samples from GSM8K dataset and save to CSV.
    
    Args:
        num_train (int): Number of examples to sample from the training set
        num_test (int): Number of examples to sample from the test set
        output_file (str, optional): Path to save the CSV file. Defaults to GSM8K_CSV_PATH.
    """
    if output_file is None:
        output_file = GSM8K_CSV_PATH
        
    print(f"Downloading {num_train} train examples and {num_test} test examples from GSM8K...")
    
    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load train and test datasets
    try:
        train_ds = load_dataset("gsm8k", "main", split="train")
        test_ds = load_dataset("gsm8k", "main", split="test")
        
        # Sample examples
        train_samples = train_ds.select(range(min(num_train, len(train_ds))))
        test_samples = test_ds.select(range(min(num_test, len(test_ds))))
        
        # Combine samples
        all_samples = []
        
        # Add train samples
        for i, sample in enumerate(train_samples):
            all_samples.append({
                "id": f"train_{i}",
                "question": sample["question"],
                "answer": sample["answer"],
                "split": "train"
            })
        
        # Add test samples
        for i, sample in enumerate(test_samples):
            all_samples.append({
                "id": f"test_{i}",
                "question": sample["question"],
                "answer": sample["answer"],
                "split": "test"
            })
        
        # Save to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["id", "question", "answer", "split", "metadata"])
            
            # Write data
            for sample in all_samples:
                metadata = json.dumps({"dataset": "gsm8k", "split": sample["split"]})
                writer.writerow([
                    sample["id"],
                    sample["question"],
                    sample["answer"],
                    sample["split"],
                    metadata
                ])
        
        print(f"Successfully saved {len(all_samples)} samples to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error downloading GSM8K samples: {e}")
        return None

if __name__ == "__main__":
    download_gsm8k_samples() 