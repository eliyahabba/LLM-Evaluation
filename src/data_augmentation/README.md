# Data Augmentation System for Language Model Testing

This system generates variations of existing datasets (SimpleQA, GSM8K, and MATH) to test language model robustness against superficial text changes.

## Overview

The data augmentation system creates variations along three main axes:

1. **Dataset-Specific Instructions** - Changes in the instructions given to the model, tailored to each dataset
2. **Textual Surface Transformations** - Adding random spaces, changing letter cases, swapping punctuation, etc.
3. **Few-shot Example Selection** - Variation in example count and selection

## System Architecture

The system consists of four main components:

1. **Data Download Module** - Downloads datasets from Hugging Face
2. **Configuration Generation Module** - Creates all possible parameter combinations
3. **Data Processing Module** - Applies transformations according to configurations
4. **Result Storage Module** - Saves configurations and processed data

## Directory Structure

```
src/data_augmentation/
├── config/
│   ├── constants.py         # Default parameters and constants
│   └── generated/           # Generated configuration files (JSON)
├── data/
│   └── processed/           # Processed data files (JSON)
├── utils/
│   ├── text_surface_augmenter.py  # Text transformation utilities
│   ├── data_downloader.py   # Dataset download utilities
│   ├── config_generator.py  # Configuration generation utilities
│   └── data_processor.py    # Data processing utilities
├── main.py                  # Main script to run the system
└── README.md                # Documentation
```

## Installation

This system requires the following dependencies:
- Python 3.7+
- Hugging Face `datasets` library
- Standard Python libraries (os, json, random, argparse, logging, itertools)
- NumPy (for text transformations)

Make sure to install the required dependencies:

```bash
pip install datasets numpy
```

## Usage

Run the main script to start the data augmentation process:

```bash
python -m src.data_augmentation.main
```

### Command-line Arguments

- `--datasets`: Datasets to process (default: all available datasets)
- `--config_dir`: Directory to store generated configurations
- `--data_dir`: Directory to store processed data
- `--cache_dir`: Directory to cache downloaded datasets
- `--limit_configs`: Limit the number of configurations to generate

Example:

```bash
python -m src.data_augmentation.main --datasets simple_qa --limit_configs 10
```

## Instruction Format

The system uses a 3-part instruction format to separate instructions from input and output formats, making it more suitable for few-shot prompting scenarios:

1. **Instruction** - The general instruction or task description (e.g., "Answer the question.")
2. **Input Format** - The format for the input/problem (e.g., "Question: {question}")
3. **Target Prefix** - The prefix for the expected answer (e.g., "Answer:")

This structure allows generating few-shot examples with consistent formatting while separating the overall task instruction from the specific input/output format. It's particularly useful for few-shot learning scenarios where you want to maintain consistent formatting across examples.

### Example Templates

For Simple QA:
```json
{
  "simple_qa": {
    "instruction": "Answer the question.",
    "input_format": "Question: {question}",
    "target_prefix": "Answer:"
  }
}
```

For GSM8K:
```json
{
  "math_standard": {
    "instruction": "",
    "input_format": "Problem: {problem}",
    "target_prefix": "Solution:"
  }
}
```

## Configuration

All default parameters and constants are defined in `config/constants.py`. Modify this file to:

- Add new datasets
- Add dataset-specific instructions
- Add new text transformation techniques
- Adjust few-shot example counts and random seed ranges

## Supported Datasets

The system currently supports the following datasets:

1. **SimpleQA** - A simple question answering dataset
2. **GSM8K** - Grade school math problems with step-by-step solutions
3. **MATH** - Advanced mathematics problems with detailed solutions

Each dataset has its own set of tailored instructions that are appropriate for the specific task.

## Transformations

The system supports the following independent text transformations:

1. **Spacing** - Adding random extra spaces between words
2. **Typos** - Introducing realistic keyboard-based typos (butter finger effect)
3. **Case Change** - Randomly changing character case
4. **Character Swap** - Swapping adjacent characters within words
5. **Punctuation Swap** - Replacing punctuation with variants

Each transformation can be applied independently, allowing you to observe the specific effect of each type of text variation on model performance. The transformations are based on the TextSurfaceAugmenter class which provides robust and customizable text modification capabilities.

## Output Format

Each generated configuration and processed data file is stored as a JSON file.

### Configuration File Example

```json
{
  "id": "001",
  "template_name": "simple_qa",
  "instruction": "Answer the question.",
  "input_format": "Question: {question}",
  "target_prefix": "Answer:",
  "text_transformations": {
    "technique": "spacing",
    "parameters": {
      "min_spaces": 1,
      "max_spaces": 3
    }
  },
  "few_shot": {
    "count": 5,
    "random_seed": 42
  },
  "dataset": "simple_qa"
}
```

### Processed Data File Example

```json
{
  "id": "simple_qa_001",
  "instruction": "Answer the question.",
  "input_format": "Question: {question}",
  "target_prefix": "Answer:",
  "question": "What might   a person do when they feel  overwhelmed?",
  "few_shot_examples": [
    {
      "question": "Question: What is the capital of France?",
      "answer": "Answer: Paris"
    },
    {
      "question": "Question: Who wrote Romeo and Juliet?",
      "answer": "Answer: William Shakespeare"
    }
  ],
  "original_question": "What might a person do when they feel overwhelmed?"
}
```

### Few-Shot Prompt Structure

The processed data is designed to be easily formatted into few-shot prompts with the following structure:

```json
{
  "question": "Answer the question.",
  "few_shot_examples": [
    {
      "question": "Question: What is the capital of France?",
      "answer": "Answer: Paris"
    },
    {
      "question": "Question: Who wrote Romeo and Juliet?",
      "answer": "Answer: William Shakespeare"
    }
  ],
  "input": "Question: What might a person do when they feel overwhelmed?",
  "target_prefix": "Answer:"
}
```

This format separates the instruction, few-shot examples, input, and target prefix, making it suitable for few-shot learning evaluation.

## Extending the System

To add new transformation techniques:
1. Add a new method to the `TextSurfaceAugmenter` class in `text_surface_augmenter.py`
2. Update the `apply_transformation` method in `data_processor.py` to handle the new technique
3. Add the new technique to the `TEXT_TRANSFORMATIONS` list in `constants.py`

To add new datasets:
1. Add the dataset to the `DATASETS` dictionary in `constants.py`
2. Add dataset-specific templates to the appropriate prompt dictionary in `constants.py`
3. Implement a prepare method in the `DataDownloader` class
4. Add any dataset-specific transformation logic in the `transform_example` method of the `DataProcessor` class

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Data Augmentation System

This system generates augmented versions of datasets for LLM evaluation, varying the following aspects:
1. **Instructions**: Different instruction templates
2. **Text Transformations**: Surface-level changes like spacing, capitalization, typos
3. **Few-shot Examples**: Different numbers and selections of examples

### Text Transformations

The system supports the following text transformations:

- **spacing**: Add random whitespace to the text
  - `probability`: Probability of replacing a space with random whitespace (0-1)
  - `min_spaces`: Minimum number of spaces to add
  - `max_spaces`: Maximum number of spaces to add
  - `word_probability`: Percentage of words to affect

- **spacing_light**: Lighter version of spacing transformation
  - Adds fewer spaces with lower probability

- **spacing_extreme**: More extreme spacing transformation
  - Adds more spaces with higher probability

- **no_spacing**: Preserves the original spacing in the text
  - No changes to spaces or whitespace

- **case_change**: Change the case of characters
  - `probability`: Probability of changing a character's case

- **typo**: Introduce typographical errors
  - `probability`: Probability of introducing a typo

- **character_swap**: Swap adjacent characters
  - `probability`: Probability of swapping characters

- **punctuation_swap**: Change or remove punctuation
  - `probability`: Probability of changing punctuation

### Datasets

The system currently supports:
- **SimpleQA**: Simple factual questions
- **GSM8K**: Grade school math problems 
- **MATH**: (disabled by default) Complex mathematical problems

### Configuration

Configurations specify the following:
- Dataset to use
- Instruction template
- Text transformation technique and parameters
- Few-shot example count and selection seed

### Output Format

The system generates output files in a JSON format with:
- Original and transformed questions
- Instruction templates
- Few-shot examples
- Metadata for tracking transformations

### Running the System

To run the system:

```bash
python main.py
```

You can modify the constants in `config/constants.py` to control:
- Available transformations and their parameters
- Instruction templates
- Few-shot example settings

### Testing

The repository includes several test scripts:
- `test_transformations.py`: Test different text transformations
- `test_spacing_parameters.py`: Test spacing parameters
- `test_new_format.py`: Test the new template format
- `test_simple_qa_csv.py`: Test loading SimpleQA from CSV
- `test_gsm8k_split.py`: Test GSM8K train/test splitting
- `test_save_examples.py`: Test saving examples to JSON
- `test_descriptive_filenames.py`: Test descriptive filename generation
- `test_no_spacing.py`: Test the no_spacing transformation option 

## Deterministic Behavior

The system now uses a global random seed to ensure deterministic behavior across runs. 
The global seed is defined in `config/constants.py` as `GLOBAL_RANDOM_SEED` (default value: 42).

This ensures that:
- Data sampling is consistent
- Few-shot example selection is consistent
- Text transformations produce the same output for the same input
- Configuration generation is deterministic

You can override the global seed when running the main script:

```
python -m src.data_augmentation.main --random_seed 123
```

All components (DataProcessor, TextSurfaceAugmenter, ConfigGenerator, etc.) use this seed
to initialize their random number generators, ensuring reproducible results. 