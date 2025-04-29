"""
Training module for the prompt dimension robustness experiment.
Handles model training and fine-tuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from src.ImproveRobustness.config import ExperimentConfig


def train_model(train_dataset, model_name=None, batch_size=None, learning_rate=None, 
             num_epochs=None, output_dir=None, fp16=None, seed=None):
    """Fine-tune the model on the prepared training data."""
    # Use defaults from ExperimentConfig if not provided
    if model_name is None:
        model_name = ExperimentConfig.MODEL_NAME
    if batch_size is None:
        batch_size = ExperimentConfig.BATCH_SIZE
    if learning_rate is None:
        learning_rate = ExperimentConfig.LEARNING_RATE
    if num_epochs is None:
        num_epochs = ExperimentConfig.NUM_TRAIN_EPOCHS
    if output_dir is None:
        output_dir = ExperimentConfig.MODELS_DIR
    if fp16 is None:
        fp16 = ExperimentConfig.FP16
    if seed is None:
        seed = ExperimentConfig.SEED
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Set up the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=ExperimentConfig.WARMUP_RATIO,
        weight_decay=ExperimentConfig.WEIGHT_DECAY,
        gradient_accumulation_steps=ExperimentConfig.GRADIENT_ACCUMULATION_STEPS,
        fp16=fp16,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_strategy="epoch",
        seed=seed
    )

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save the model
    trained_model_dir = output_dir / "trained_model"
    model.save_pretrained(trained_model_dir)
    tokenizer.save_pretrained(trained_model_dir)

    return model, tokenizer 