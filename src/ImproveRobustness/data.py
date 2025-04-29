"""
Data handling module for the prompt dimension robustness experiment.
Handles processing of datasets for model training and evaluation.
"""

from datasets import Dataset
from src.ImproveRobustness.config import ExperimentConfig
from src.ImproveRobustness.prepare_data import load_data


def prepare_datasets_for_training(data_dir=None):
    """
    Load prepared data and convert to HuggingFace datasets for training.
    
    Args:
        data_dir (Path, optional): Directory containing processed data.
            If None, uses ExperimentConfig.DATA_DIR
    
    Returns:
        Dataset: Processed dataset ready for model training
    """
    # Load prepared data
    train_data, _ = load_data(data_dir=data_dir)
    
    # Function to prepare input for the model
    def prepare_inputs(examples):
        inputs = examples["raw_input"]
        outputs = examples["generated_text"]

        # Format as instruction tuning examples
        formatted_examples = []
        for input_text, output_text in zip(inputs, outputs):
            formatted_examples.append(f"{input_text}{output_text}")

        return {"text": formatted_examples}

    # Create datasets
    train_dataset = Dataset.from_pandas(train_data)
    processed_train_dataset = train_dataset.map(
        prepare_inputs,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    return processed_train_dataset 