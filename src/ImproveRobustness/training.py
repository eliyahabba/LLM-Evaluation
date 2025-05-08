import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import wandb
from colorama import init, Fore, Style
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer

from src.ImproveRobustness.config import ExperimentConfig

load_dotenv()

# Initialize colorama
init(autoreset=True)  # This will auto-reset colors after each print

class Trainer:
    def __init__(
            self,
            model_name=ExperimentConfig.MODEL_NAME,
            train_data_path=None,
            eval_data_path=None,
            output_dir=None,
            access_token=ExperimentConfig.ACCESS_TOKEN,
            quant_config_type=ExperimentConfig.QUANT_CONFIG_TYPE,
            use_lora=ExperimentConfig.USE_LORA,
            target_dimension=ExperimentConfig.TARGET_DIMENSION,
            training_dimension_values=None,
            excluded_dimension_values=None
    ):
        self.model_name = model_name
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.quant_config_type = quant_config_type
        self.access_token = access_token
        self.use_lora = use_lora
        self.target_dimension = target_dimension
        self.training_dimension_values = training_dimension_values or ExperimentConfig.TRAINING_DIMENSION_VALUES
        self.excluded_dimension_values = excluded_dimension_values or ExperimentConfig.EXCLUDED_DIMENSION_VALUES

        # Set default output directory if not provided
        if output_dir is None:
            self.output_dir = ExperimentConfig.RESULTS_DIR
        else:
            self.output_dir = Path(output_dir)

        self.models_output_dir = self.output_dir / "models"
        self.models_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize HuggingFace login if token provided
        if self.access_token:
            login(token=self.access_token)

        # Load HF token from env if not provided
        if not access_token:
            access_token = os.environ.get("HF_ACCESS_TOKEN")
        self.access_token = access_token
        print(f"{Fore.BLUE}Using HuggingFace access token: {self.access_token}")
        if self.access_token:
            login(token=self.access_token)

    def _load_tokenizer(self):
        """Load tokenizer for the specified model."""
        print(f"{Fore.BLUE}Loading tokenizer for {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _get_best_gpu(self):
        """Get GPU with most available memory."""
        # Check for MPS (Mac GPU) first
        if torch.backends.mps.is_available():
            print("\nMacOS GPU (MPS) available")
            print("Using Metal Performance Shaders for GPU acceleration")
            return "auto"  # Changed from "mps" to "auto" for better compatibility

        # Check for CUDA GPUs
        if not torch.cuda.is_available():
            print("No GPU available, using CPU")
            return "auto"

        # Get the number of GPUs
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            print("No GPU available, using CPU")
            return "auto"

        # Print info about all available GPUs
        print("\nAvailable CUDA GPUs:")
        for gpu_id in range(n_gpus):
            props = torch.cuda.get_device_properties(gpu_id)
            total_memory = props.total_memory / 1024 ** 3
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3
            free_memory = total_memory - allocated_memory
            print(f"GPU {gpu_id}: {props.name}")
            print(f"    Total Memory: {total_memory:.2f}GB")
            print(f"    Used Memory: {allocated_memory:.2f}GB")
            print(f"    Free Memory: {free_memory:.2f}GB")

        # Find GPU with most free memory
        max_free_memory = 0
        best_gpu = 0

        for gpu_id in range(n_gpus):
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id

        print(f"\n>>> Selected GPU {best_gpu} ({torch.cuda.get_device_properties(best_gpu).name})")
        print(f">>> Available Memory: {max_free_memory / 1024 ** 3:.2f}GB")
        return {"": best_gpu}

    def _setup_quantization(self):
        """Set up quantization configuration based on the specified type."""
        if self.quant_config_type is None:
            return None

        quantization_configs = {
            "4bit": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": False
            },
            "8bit": {
                "load_in_8bit": True,
                "bnb_8bit_quant_type": "nf8",
                "bnb_8bit_compute_dtype": torch.float16,
                "bnb_8bit_use_double_quant": False
            }
        }

        config_params = quantization_configs.get(self.quant_config_type)
        if config_params:
            return BitsAndBytesConfig(**config_params)

        raise ValueError(f"Unsupported quantization type: {self.quant_config_type}. "
                         f"Supported types are: {list(quantization_configs.keys())}")

    def create_versioned_id(self, base_id):
        """Create a versioned identifier if the original already exists."""
        dir_path = self.models_output_dir / base_id

        # If path doesn't exist, return base_id
        if not os.path.exists(dir_path):
            return base_id

        # Find available version number
        version = 2
        while True:
            versioned_id = f"{base_id}_v{version}"
            versioned_path = os.path.join(self.models_output_dir, versioned_id)

            if not os.path.exists(versioned_path):
                return versioned_id

            version += 1

    def _validate_data_format(self, data_path):
        """Validate that the data file has the required columns."""
        # Skip validation as data format is confirmed by user
        return True

    def prepare_dataset(self, data_path, include_completion=True, limit=None, start_idx=None, end_idx=None):
        """Prepare dataset for training from the dataframe."""
        # Load data directly without validation
        df = pd.read_parquet(data_path)

        if limit is not None:
            df = df.head(limit)

        if start_idx is not None and end_idx is not None:
            df = df[start_idx:end_idx]

        # # Filter by dimension value if specified (we know dimension exists)
        # if self.training_dimension_values and self.target_dimension:
        #     # For training data, only use specified dimension values
        #     df = df[df[self.target_dimension].isin(self.training_dimension_values)]

        # Create dataset
        dataset = Dataset.from_pandas(df)

        return dataset

    def create_chat_dataset(self, dataset, tokenizer, include_completion=True):
        """Convert dataset to chat format using tokenizer's chat template."""

        def format_chat(example):
            # The prompt is already in the raw_input field
            prompt = example["raw_input"]

            messages = [{"role": "user", "content": prompt}]

            # Add completion if needed
            if include_completion and "ground_truth" in example:
                messages.append({"role": "assistant", "content": example["ground_truth"]})

            # Check if tokenizer supports chat templates
            if hasattr(tokenizer, 'apply_chat_template'):
                # Apply chat template
                chat_str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=not include_completion
                )
            else:
                if include_completion and "ground_truth" in example:
                    chat_str = f"{prompt}{example['ground_truth']}"
                else:
                    chat_str = prompt
                    # Return a dict to update the 'text' column
            return {"text": chat_str}

        # Apply formatting
        chat_data = dataset.map(format_chat)

        # Convert to text-only dataset
        return Dataset.from_dict({"text": chat_data["text"]})
    def create_chat_dataset(self, dataset, tokenizer, include_completion=True):
        """Convert dataset to chat format using tokenizer's chat template."""

        def format_chat(example):
            # The prompt is already in the raw_input field
            prompt = example["raw_input"]

            messages = [{"role": "user", "content": prompt}]

            # Add completion if needed
            if include_completion and "ground_truth" in example:
                messages.append({"role": "assistant", "content": example["ground_truth"]})

            # Check if tokenizer supports chat templates
            if include_completion and "ground_truth" in example:
                chat_str = f"### {prompt}\n{example['ground_truth']}"
            else:
                chat_str = prompt
                # Return a dict to update the 'text' column
            return {"text": chat_str}

        # Apply formatting
        chat_data = dataset.map(format_chat)

        # Convert to text-only dataset
        return Dataset.from_dict({"text": chat_data["text"]})

    def run_pipeline(self,
                     eval_before_finetuning=ExperimentConfig.EVAL_BEFORE_FINETUNING,
                     do_finetuning=ExperimentConfig.DO_FINETUNING,
                     eval_after_finetuning=ExperimentConfig.EVAL_AFTER_FINETUNING):
        """Run the complete training pipeline."""
        # Set up training parameters
        learning_rate = ExperimentConfig.LEARNING_RATE
        logging_steps = ExperimentConfig.LOGGING_STEPS

        # Create unique ID for this run
        current_time = pd.Timestamp.now().strftime("%m%d")
        bit_info = f"{self.quant_config_type}-bit" if self.quant_config_type else "None-bit"
        base_id = f"{current_time}_{self.model_name.split('/')[-1]}--{bit_info}--lr-{learning_rate}--log_steps-{logging_steps}"
        id_prompt_name = self.create_versioned_id(base_id)
        print(f"{Fore.GREEN}Run ID: {id_prompt_name}")

        # Define paths
        model_output_path = self.models_output_dir / id_prompt_name

        # Load tokenizer
        tokenizer = self._load_tokenizer()

        # Prepare datasets
        print(f"{Fore.GREEN}Preparing datasets...")
        train_dataset_raw = self.prepare_dataset(self.train_data_path,
                                                 start_idx=0,
                                                 end_idx=100,
                                                 include_completion=True
                                                 )
        test_dataset_raw = self.prepare_dataset(
            self.eval_data_path,
            start_idx=0,
            end_idx=25,
            include_completion=False
        )
        val_dataset_raw = self.prepare_dataset(
            self.eval_data_path,
            start_idx=25,
            end_idx=50
        )

        # Convert to chat format
        train_dataset = self.create_chat_dataset(train_dataset_raw, tokenizer)
        test_dataset = self.create_chat_dataset(test_dataset_raw, tokenizer, include_completion=False)
        val_dataset = self.create_chat_dataset(val_dataset_raw, tokenizer)

        # Print dataset sizes
        print(f"{Fore.CYAN}Train set size: {len(train_dataset)}")
        print(f"{Fore.CYAN}Test set size: {len(test_dataset)}")
        print(f"{Fore.CYAN}Validation set size: {len(val_dataset)}")

        # For evaluation before fine-tuning
        train_subset_raw = self.prepare_dataset(self.train_data_path, limit=10, include_completion=False)
        train_subset = self.create_chat_dataset(train_subset_raw, tokenizer, include_completion=False)

        # Get best GPU
        device_map = self._get_best_gpu()
        if isinstance(device_map, dict):
            gpu_id = device_map.get("", 0)
        else:
            gpu_id = 0  # Default to first GPU if device_map is a string

        # Clean GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Setup quantization
        quant_config = self._setup_quantization()

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True
        )

        # Evaluate before fine-tuning if requested
        if eval_before_finetuning:
            self._eval_before_finetuning(model, tokenizer, test_dataset, model_output_path)

        # Fine-tune model if requested
        if do_finetuning:
            # LoRA configuration if needed
            peft_config = None
            if self.use_lora:
                print("Using LoRA for fine-tuning")
                peft_config = LoraConfig(
                    r=ExperimentConfig.LORA_R,
                    lora_alpha=ExperimentConfig.LORA_ALPHA,
                    lora_dropout=ExperimentConfig.LORA_DROPOUT,
                    bias="none",
                    task_type="CAUSAL_LM",
                    inference_mode=False,
                    modules_to_save=None
                )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=model_output_path,
                num_train_epochs=ExperimentConfig.NUM_TRAIN_EPOCHS,
                per_device_train_batch_size=ExperimentConfig.PER_DEVICE_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=ExperimentConfig.GRADIENT_ACCUMULATION_STEPS,
                optim="paged_adamw_32bit",
                save_steps=ExperimentConfig.SAVE_STEPS,
                logging_steps=logging_steps,
                learning_rate=learning_rate,
                weight_decay=ExperimentConfig.WEIGHT_DECAY,
                fp16=ExperimentConfig.FP16,
                bf16=False,
                max_grad_norm=ExperimentConfig.MAX_GRAD_NORM,
                max_steps=-1,
                warmup_ratio=ExperimentConfig.WARMUP_RATIO,
                group_by_length=True,
                lr_scheduler_type=ExperimentConfig.LR_SCHEDULER_TYPE,
                eval_steps=ExperimentConfig.LOGGING_STEPS,
                eval_strategy="steps",
                per_device_eval_batch_size=ExperimentConfig.PER_DEVICE_EVAL_BATCH_SIZE,
                save_total_limit=ExperimentConfig.SAVE_TOTAL_LIMIT,
                report_to="wandb"
            )

            # Initialize wandb if needed and available
            if self._is_wandb_available():
                try:
                    # Define wandb project name
                    wandb_project = "ImproveRobustness"

                    # Initialize wandb
                    wandb_run = wandb.init(
                        project=wandb_project,
                        id=f"{id_prompt_name}_gpu{gpu_id}",
                        config={
                            "learning_rate": training_args.learning_rate,
                            "epochs": training_args.num_train_epochs,
                            "logging_steps": training_args.logging_steps,
                            "per_device_train_batch_size": training_args.per_device_train_batch_size,
                            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                            "weight_decay": training_args.weight_decay,
                            "warmup_ratio": training_args.warmup_ratio,
                            "lr_scheduler_type": training_args.lr_scheduler_type,
                            "optim": training_args.optim,
                            "model_name": self.model_name,
                            "peft_method": "LoRA" if self.use_lora else "None",
                            "quantization": self.quant_config_type if self.quant_config_type else "None",
                            "gpu_id": gpu_id,
                            "dimension": self.target_dimension,
                            "training_values": self.training_dimension_values,
                            "excluded_values": self.excluded_dimension_values,
                            "train_dataset_size": len(train_dataset),
                            "eval_dataset_size": len(val_dataset),
                            "test_dataset_size": len(test_dataset),
                            "run_id": id_prompt_name
                        },
                    )

                    # Print wandb information
                    print(f"{Fore.BLUE}WandB Project: {wandb_project}")
                    print(f"{Fore.BLUE}WandB Run ID: {wandb_run.id}")
                    print(f"{Fore.BLUE}WandB URL: {wandb_run.get_url()}")

                    # Log dataset samples for reference
                    train_examples = train_dataset[:3]["text"] if len(train_dataset) >= 3 else train_dataset["text"]
                    wandb.log({"train_examples": wandb.Table(data=[[ex] for ex in train_examples], columns=["text"])})

                except Exception as e:
                    print(f"{Fore.YELLOW}Warning: Failed to initialize wandb: {e}")

            # Initialize trainer
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                peft_config=peft_config,
                args=training_args,
            )

            # Clean GPU cache
            torch.cuda.empty_cache()

            # Move model to selected GPU
            if torch.cuda.is_available():
                trainer.model = trainer.model.to(f"cuda:{gpu_id}")

            # Display sample training data
            print(f"\n{Fore.GREEN}{'='*30} SAMPLE TRAINING DATA {'='*30}")
            print(f"{Fore.GREEN}Showing {min(5, len(train_dataset))} examples from the training dataset")
            print(f"{Fore.GREEN}{'='*80}\n")

            print(f"{Fore.GREEN}Starting training for model: {self.model_name}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Training data: {self.train_data_path}")
            print(f"{Fore.CYAN}Validation data: {self.eval_data_path}")
            print(f"{Fore.CYAN}Output directory: {model_output_path}")

            print(f"{Fore.GREEN}Training started...")
            trainer.train()
            print(f"{Fore.MAGENTA}Training finished.")

            # Save the model
            trainer.model.save_pretrained(model_output_path)
            trainer.tokenizer.save_pretrained(model_output_path)
            print(f"{Fore.MAGENTA}Model saved to {model_output_path}")

            # Finish wandb run


            # Evaluate on validation set
            print(f"{Fore.GREEN}Evaluating on validation set...")
            eval_results = trainer.evaluate()
            print(f"{Fore.MAGENTA}Evaluation finished.")
            print(f"{Fore.CYAN}Eval results: {eval_results}")

            # Log metrics to wandb with prefix to distinguish from training metrics
            if self._is_wandb_available():
                # Add prefix to evaluation metrics to distinguish them from training metrics
                prefixed_eval_results = {f"eval/{k}": v for k, v in eval_results.items()}

                # Add some extra useful metrics
                prefixed_eval_results.update({
                    "train/epoch": trainer.state.epoch,
                    "train/total_steps": trainer.state.global_step,
                    "train/learning_rate": trainer.state.learning_rate if hasattr(trainer.state, "learning_rate") else learning_rate,
                    "eval/timestamp": pd.Timestamp.now().isoformat(),
                })

                # Log the metrics
                wandb.log(prefixed_eval_results)

        # Evaluate after fine-tuning if requested
        if eval_after_finetuning:
            if not do_finetuning:
                # If we didn't fine-tune, use the path to an existing model
                model_output_path = self.models_output_dir / "previous_model"

            self._eval_after_finetuning(
                model_path=model_output_path,
                tokenizer=tokenizer,
                test_dataset=test_dataset,
                train_subset=train_subset
            )
        if self._is_wandb_available():
            try:
                wandb.finish()
            except:
                pass
        return model_output_path

    def _is_wandb_available(self):
        """Check if wandb is available and configured."""
        try:
            import wandb
            if wandb.api.api_key is None:
                print(f"{Fore.YELLOW}Warning: wandb is installed but not logged in. Run 'wandb login' first.")
                return False
            return True
        except ImportError:
            print(f"{Fore.YELLOW}Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")
            return False
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error checking wandb availability: {e}")
            return False

    def _eval_before_finetuning(self, model, tokenizer, test_dataset, output_path):
        """Evaluate model before fine-tuning."""
        print(f"{Fore.GREEN}Evaluating model before fine-tuning...")
        results_dir = self.output_dir / "evaluations"
        results_dir.mkdir(exist_ok=True, parents=True)

        # Here you'd implement evaluation logic
        # This is a placeholder that would need to be customized
        predictions = self._generate_predictions(model, tokenizer, test_dataset)

        # Save predictions
        test_results_path = results_dir / f"{output_path.name}_results_before_finetuning.csv"
        predictions.to_csv(test_results_path, index=False)
        print(f"{Fore.MAGENTA}Pre-training evaluation saved to {test_results_path}")

    def _eval_after_finetuning(self, model_path, tokenizer, test_dataset, train_subset):
        """Evaluate model after fine-tuning."""
        print(f"\n{Fore.GREEN}{'*'*30} STARTING POST-TRAINING EVALUATION {'*'*30}")
        print(f"{Fore.GREEN}Evaluating model: {model_path}")
        print(f"{Fore.GREEN}{'*'*80}\n")

        results_dir = self.output_dir / "evaluations"
        results_dir.mkdir(exist_ok=True, parents=True)

        # Load fine-tuned model
        model = self._load_finetuned_model(model_path)

        # Evaluate on test set
        print(f"{Fore.GREEN}{'#'*30} EVALUATING ON TEST SET {'#'*30}")
        test_predictions = self._generate_predictions(model, tokenizer, test_dataset)
        test_results_path = results_dir / f"{model_path.name}_results_after_finetuning.csv"
        test_predictions.to_csv(test_results_path, index=False)
        print(f"{Fore.MAGENTA}Test set evaluation saved to: {test_results_path}")

        # Evaluate on training subset
        print(f"{Fore.GREEN}{'#'*30} EVALUATING ON TRAINING SUBSET {'#'*30}")
        train_predictions = self._generate_predictions(model, tokenizer, train_subset)
        train_results_path = results_dir / f"{model_path.name}_train_results_after_finetuning.csv"
        train_predictions.to_csv(train_results_path, index=False)
        print(f"{Fore.MAGENTA}Training subset evaluation saved to: {train_results_path}")

        # Print summary of evaluation
        print(f"\n{Fore.GREEN}{'*'*30} EVALUATION SUMMARY {'*'*30}")
        print(f"{Fore.CYAN}Total test examples evaluated: {len(test_predictions)}")
        print(f"{Fore.CYAN}Total training examples evaluated: {len(train_predictions)}")
        print(f"{Fore.CYAN}Results saved to: {results_dir}")
        print(f"{Fore.GREEN}{'*'*80}\n")

        # Log post-training evaluation to wandb if available
        if self._is_wandb_available():
            try:
                # Calculate some basic metrics (this is simplified - you may want to add more)
                metrics = {
                    "final_eval/test_samples": len(test_predictions),
                    "final_eval/train_samples": len(train_predictions),
                    "final_eval/model_path": str(model_path),
                }

                # Log a sample of predictions
                test_table_data = []
                for i, row in test_predictions.head(5).iterrows():
                    test_table_data.append([row["input"], row["prediction"]])

                wandb.log({
                    **metrics,
                    "final_eval/test_predictions": wandb.Table(
                        data=test_table_data,
                        columns=["input", "prediction"]
                    )
                })
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Failed to log post-training evaluation to wandb: {e}")

    def _load_finetuned_model(self, model_path):
        """Load the fine-tuned model."""
        torch.cuda.empty_cache()

        if self.use_lora:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map="balanced",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            return model.merge_and_unload()
        else:
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="balanced",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

    def _generate_predictions(self, model, tokenizer, dataset):
        """Generate predictions for a dataset."""
        # This is a simplified implementation - you may need to customize
        model.eval()
        predictions = []

        print(f"\n{Fore.GREEN}{'='*80}")
        print(f"{Fore.GREEN}Generating predictions for {len(dataset)} examples...")
        print(f"{Fore.GREEN}{'='*80}\n")

        batch_size = 1  # Process one example at a time to manage memory
        for i, item in enumerate(dataset):
            try:
                # The input should already be in the correct format from create_chat_dataset
                inputs = tokenizer(item["text"], return_tensors="pt", padding=True).to(model.device)
                # Handle very long inputs by truncating if needed
                max_length = tokenizer.model_max_length
                if inputs["input_ids"].shape[1] > max_length:
                    print(f"{Fore.YELLOW}Warning: Input {i} exceeds max length, truncating")
                    inputs["input_ids"] = inputs["input_ids"][:, :max_length]
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]

                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pad_token_id=tokenizer.pad_token_id,
                        max_new_tokens=ExperimentConfig.MAX_NEW_TOKENS,
                        do_sample=ExperimentConfig.DO_SAMPLE,
                        # Explicitly set these to None to prevent warnings
                        temperature=None,
                        top_p=None,
                        top_k=None
                    )

                # Decode generated text
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Get the response only (remove the prompt)
                response = generated_text[len(item["text"]):]

                # Add to predictions
                predictions.append({
                    "index": i,
                    "input": item["text"],
                    "prediction": response.strip()
                })

                # Print the example and prediction with nice formatting
                print(f"{Fore.CYAN}{'='*40} Example {i+1}/{len(dataset)} {'='*40}")
                print(f"{Fore.BLUE}INPUT:")
                print(f"{Fore.WHITE}{item['text'][:500]}{'...' if len(item['text']) > 500 else ''}")
                print(f"\n{Fore.MAGENTA}MODEL OUTPUT:")
                print(f"{Fore.WHITE}{response.strip()}")
                print(f"{Fore.CYAN}{'='*90}\n")

                # Clean up memory for CUDA tensors
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"{Fore.RED}Error generating prediction for example {i}: {e}")
                predictions.append({
                    "index": i,
                    "input": item["text"],
                    "prediction": f"ERROR: {str(e)}"
                })

        return pd.DataFrame(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for dimension robustness")

    # Parameters for data paths - make optional with defaults from config
    parser.add_argument("--model_name", type=str, default=ExperimentConfig.MODEL_NAME,
                        help="Name of the model to fine-tune")
    parser.add_argument("--train_data_path", type=str,
                        default=str(ExperimentConfig.DATA_DIR / "train_data.parquet"),
                        help="Path to training data in parquet format")
    parser.add_argument("--eval_data_path", type=str,
                        default=str(ExperimentConfig.DATA_DIR / "test_data.parquet"),
                        help="Path to evaluation data in parquet format")

    # Optional parameters
    parser.add_argument("--output_dir", type=str, default=str(ExperimentConfig.RESULTS_DIR),
                        help="Directory to save results")
    parser.add_argument("--access_token", type=str, default=ExperimentConfig.ACCESS_TOKEN,
                        help="HuggingFace access token")
    parser.add_argument("--quant_config_type", type=str, choices=[None, "4bit", "8bit"],
                        default=ExperimentConfig.QUANT_CONFIG_TYPE,
                        help="Quantization configuration type")
    parser.add_argument("--use_lora", action="store_true", default=ExperimentConfig.USE_LORA,
                        help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--target_dimension", type=str, default=ExperimentConfig.TARGET_DIMENSION,
                        help="Target dimension to test robustness for")
    parser.add_argument("--eval_before_finetuning", action="store_true",
                        default=ExperimentConfig.EVAL_BEFORE_FINETUNING,
                        help="Evaluate model before fine-tuning")
    parser.add_argument("--do_finetuning", action="store_true", default=ExperimentConfig.DO_FINETUNING,
                        help="Perform fine-tuning")
    parser.add_argument("--eval_after_finetuning", action="store_true", default=ExperimentConfig.EVAL_AFTER_FINETUNING,
                        help="Evaluate model after fine-tuning")

    args = parser.parse_args()

    # Validate that data files exist before creating trainer
    train_path = Path(args.train_data_path)
    eval_path = Path(args.eval_data_path)

    if not train_path.exists():
        print(f"{Fore.RED}Error: Training data file not found: {train_path}")
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not eval_path.exists():
        print(f"{Fore.RED}Error: Testing data file not found: {eval_path}")
        raise FileNotFoundError(f"Testing data file not found: {eval_path}")

    # Set random seed for reproducibility
    ExperimentConfig.set_seed()
    print(f"{Fore.GREEN}Starting training pipeline with seed: {ExperimentConfig.SEED}")

    # Create trainer
    trainer = Trainer(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        output_dir=args.output_dir,
        access_token=args.access_token,
        quant_config_type=args.quant_config_type,
        use_lora=args.use_lora,
        target_dimension=args.target_dimension
    )

    # Run training pipeline
    trainer.run_pipeline(
        eval_before_finetuning=args.eval_before_finetuning,
        do_finetuning=args.do_finetuning,
        eval_after_finetuning=args.eval_after_finetuning
    )
