# LLM-Dataset-Fix# Robust LLM Evaluation

## Overview
This project aims to evaluate the robustness of Language Model Models (LLMs)
by modifying datasets and assessing model performance. 
It addresses challenges in evaluating LLMs, particularly focusing on issues
related to dataset biases and model memorization.

## Project Structure
The project is organized as follows:

- `data/`: Contains original and modified datasets.
- `models/`: Contains pretrained LLM models.
- `Modifiers/`: Contains scripts for modifying datasets.
- `experiments/`: Contains scripts for running experiments and analyzing results.
- `README.md`: Documentation explaining the project and its structure.

## Dataset Modification
Different types of dataset modifications are implemented based on specific tasks:
- `MultipleChoiceModifier`: Shuffles answer choices for multiple-choice questions.
- `QAModifier`: Implements modifications for question-answering datasets.
- `NLIModifier`: Implements modifications for natural language inference datasets.

## Experimentation
The project conducts experiments to evaluate LLM performance:
1. Identifying tasks and models for evaluation.
2. Modifying datasets using appropriate modifiers.
3. Running models on both original and modified datasets.
4. Analyzing results to assess model robustness.

## Usage
1. Clone the repository: `git clone https://github.com/your-username/robust-llm-evaluation.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run dataset modifications: `python dataset_modification/modify_dataset.py`
4. Run experiments: `python experiments/run_experiment.py`
5. Analyze results: `python experiments/analyze_results.py`

## Contributors

## License
This project is licensed under the [MIT License](LICENSE).
