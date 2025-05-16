"""
Constants and default parameters for the data augmentation system.
"""
from pathlib import Path

# Global random seed for deterministic behavior
GLOBAL_RANDOM_SEED = 42

# Dataset constants
DATASETS = {
    "simple_qa": "local_csv",  # Loaded from local CSV file with train/test split
    "gsm8k": "local_csv",      # Loaded from local CSV file: src/data_augmentation/data/gsm8k_samples.csv
    # MATH dataset is disabled for performance reasons
    # "math": "competition_math"  
}

# Maximum token count for answers - filter examples with more tokens than these values
# None means no filtering for that dataset
MAX_ANSWER_TOKENS = {
    "simple_qa": None,   # Filter SimpleQA examples with more than 50 tokens
    "gsm8k": 553,      # Filter GSM8K examples with more than 200 tokens
    "math": None       # No filtering for MATH dataset
}

# Number of examples to sample per configuration
EXAMPLES_PER_CONFIG = {
    "simple_qa": 1000,
    "gsm8k": 500
}

# Maximum configurations per dataset
MAX_CONFIGS_PER_DATASET = 100
# simple_qa= [
#     "Question: {question}\nAnswer:",
#     "Q: {question}\nA:",
#     "Question:\n{question}\nAnswer:\n",
#     "Answer the question.\nQuestion: {question} Answer:",
#     "Answer the question.\nQuestion:\n{question}Answer:\n"
#     "{question}",
#     "You are an assistant for question-answering tasks. Question: {question} Answer:"
#     "Below is a question for you to answer. Please provide a detailed response. {question}"
#     "Instruction: Answer the following question.\nQuestion: {question}\nAnswer:"
#     "Could you please answer this question?\n\n{question}\nAnswer:",
#     "Provide an answer to the question below:\n\nQuestion:\n{question}\n\nAnswer:",
#     "Please answer the following question:\n{question}}\nAnswer:"
# ],
#
# # Common templates for math-related datasets (GSM8K and MATH)
# MATH_TEMPLATES = [
#     "Given a mathematics problem, determine the answer. Simplify your answer as much as possible.\nProblem: {problem}\nAnswer:",
#     "Problem:\n{problem}\nSolution:\n",
#     "{problem}\nSolution:",
#     "Problem: {problem}\nSolution:",
#     "Could you provide a solution to the following question: {problem}\n",
#     "Please provide a solution to the following problems:\n{problem}\n",
#     "Please address the following problems:\n{problem}",
#     "You are a very helpful mathematical problem solver. Please provide a solution to the following questions: {problem}\n",
#     "As an AI expert in math, could you help me to answer the problem below:\n{problem}\nSolution:\n",
#     "As a helpful Artificial Intelligence Assistant, please answer the following question.\n{problem}\n",
#     "Solve the following problem: {problem}\nPut your answer on its own line after \"Final Answer:\"",
#     "Please answer the following question:\n{problem}\nInclude your answer in the line after \"Final Answer:\"",
#     "Please help me to address the following question:\n{problem}\nInclude your answer in the line after \"Final Answer:\""
# ]
math_prompts = {
    "math_simplify": {
        "instruction": "Given a mathematics problem, determine the answer. Simplify your answer as much as possible.\n\n",
        "input_format": "Problem: {problem}",
        "target_prefix": "Answer: "
    },
    "math_multiline": {
        "instruction": "",
        "input_format": "Problem:\n{problem}",
        "target_prefix": "Solution:\n"
    },
    "math_basic": {
        "instruction": "",
        "input_format": "{problem}",
        "target_prefix": "Solution: "
    },
    "math_standard": {
        "instruction": "",
        "input_format": "Problem: {problem}",
        "target_prefix": "Solution: "
    },
    "math_polite_request": {
        "instruction": "Could you provide a solution to the following question: ",
        "input_format": "{problem}",
        "target_prefix": "\n"
    },
    "math_plural_request": {
        "instruction": "Please provide a solution to the following problems:\n",
        "input_format": "{problem}",
        "target_prefix": "\n"
    },
    "math_address_request": {
        "instruction": "Please address the following problems:\n",
        "input_format": "{problem}",
        "target_prefix": ""
    },
    "math_expert_solver": {
        "instruction": "You are a very helpful mathematical problem solver. Please provide a solution to the following questions: ",
        "input_format": "{problem}",
        "target_prefix": "\n"
    },
    "math_ai_expert": {
        "instruction": "As an AI expert in math, could you help me to answer the problem below:\n\n",
        "input_format": "{problem}\n",
        "target_prefix": "Solution:\n"
    },
    "math_helpful_assistant": {
        "instruction": "As a helpful Artificial Intelligence Assistant, please answer the following question.\n",
        "input_format": "{problem}\n",
        "target_prefix": ""
    },
    "math_final_answer": {
        "instruction": "Solve the following problem: ",
        "input_format": "{problem}\n",
        "target_prefix": "Put your answer on its own line after \"Final Answer:\""
    },
    "math_final_answer_please": {
        "instruction": "Please answer the following question:\n",
        "input_format": "{problem}\n",
        "target_prefix": "Include your answer in the line after \"Final Answer:\""
    },
    "math_help_address": {
        "instruction": "Please help me to address the following question:\n",
        "input_format": "{problem}\n",
        "target_prefix": "Include your answer in the line after \"Final Answer:\""
    }
}


qa_prompts = {
    "simple_qa": {
        "instruction": "Answer the question.\n\n",
        "input_format": "Question: {question}",
        "target_prefix": "Answer: "
    },
    "minimal_qa": {
        "instruction": "",
        "input_format": "Q: {question}",
        "target_prefix": "A: "
    },
    "multiline_qa": {
        "instruction": "",
        "input_format": "Question:\n{question}",
        "target_prefix": "Answer:\n"
    },
    "instructed_qa": {
        "instruction": "Answer the question: ",
        "input_format": "Question: {question}",
        "target_prefix": "Answer: "
    },
    "instructed_multiline_qa": {
        "instruction": "Answer the question.\n",
        "input_format": "Question:\n{question}",
        "target_prefix": "Answer:\n"
    },
    "raw_question": {
        "instruction": "",
        "input_format": "{question}",
        "target_prefix": ""
    },
    "assistant_qa": {
        "instruction": "You are an assistant for question-answering tasks.\n\n",
        "input_format": "Question: {question}",
        "target_prefix": "Answer: "
    },
    "detailed_response": {
        "instruction": "Below is a question for you to answer. Please provide a detailed response.\n\n",
        "input_format": "{question}",
        "target_prefix": ""
    },
    "instruction_qa": {
        "instruction": "Answer the following question: ",
        "input_format": "Question: {question}",
        "target_prefix": "Answer: "
    },
    "polite_qa": {
        "instruction": "Could you please answer this question?\n\n",
        "input_format": "{question}\n",
        "target_prefix": "Answer: "
    },
    "formal_qa": {
        "instruction": "Provide an answer to the question below:\n\n",
        "input_format": "Question:\n{question}\n\n",
        "target_prefix": "Answer: "
    },
    "simple_request_qa": {
        "instruction": "Please answer the following question:\n",
        "input_format": "{question}\n",
        "target_prefix": "Answer: "
    }
}

# Dataset-specific instruction variations
DATASET_INSTRUCTIONS = {
    # SimpleQA instructions - templates where {problem} will be replaced with the actual question
    "simple_qa": qa_prompts,
    # GSM8K instructions - using the same templates as MATH
    "gsm8k": math_prompts,
    
    # MATH instructions - templates where {problem} will be replaced with the actual problem
    # Kept for reference but disabled for now
    "math": math_prompts
}


# # Textual transformation techniques
# TEXT_TRANSFORMATIONS = [
#     {
#         "name": "spacing",
#         "params": {
#             # min_spaces=0 means don't add spaces (keep original text)
#             # min_spaces>0 means add random spaces
#             "min_spaces": 0,  # Set to 0 to keep original text without spaces
#             "max_spaces": 3
#         }
#     },
#     {
#         "name": "typo",
#         "params": {
#             "probability": 0.05  # Used by the butter_finger method
#         }
#     },
#     {
#         "name": "case_change",
#         "params": {
#             "probability": 0.1  # Used by the change_char_case method
#         }
#     },
#     {
#         "name": "character_swap",
#         "params": {
#             "probability": 0.1  # Used by the swap_characters method
#         }
#     },
#     {
#         "name": "punctuation_swap",
#         "params": {
#             "probability": 0.2  # Used by the switch_punctuation method
#         }
#     }
# ]
TEXT_TRANSFORMATIONS = [
    # {
    #     "name": "spacing",
    #     "params": {
    #         "probability": 0.8,
    #         "min_spaces": 1,
    #         "max_spaces": 3,
    #         "word_probability": 0.5
    #     }
    # },
    # {
    #     "name": "spacing_light",
    #     "params": {
    #         "probability": 0.6,
    #         "min_spaces": 1,
    #         "max_spaces": 2,
    #         "word_probability": 0.2
    #     }
    # },
    # {
    #     "name": "spacing_extreme",
    #     "params": {
    #         "probability": 0.9,
    #         "min_spaces": 2,
    #         "max_spaces": 5,
    #         "word_probability": 0.8
    #     }
    # },
    # {
    #     "name": "no_spacing",
    #     "params": {
    #         "preserve_original": True
    #     }
    # },
    {
        "name": "no_transform",
        "params": {
            "disable_transformations": True
        }
    },
    # {
    #     "name": "case_change",
    #     "params": {
    #         "probability": 0.1
    #     }
    # },
    # {
    #     "name": "typo",
    #     "params": {
    #         "probability": 0.05
    #     }
    # },
    # {
    #     "name": "character_swap",
    #     "params": {
    #         "probability": 0.08
    #     }
    # },
    # {
    #     "name": "punctuation_swap",
    #     "params": {
    #         "probability": 0.1
    #     }
    # }
]
# Few-shot example parameters
FEW_SHOT = {
    "count_range": [5],
    "seed_range": list(range(1, 11))  # Seeds 1-10
}

# File paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset file paths
SIMPLE_QA_CSV = "simple_qa_with_split.csv"
GSM8K_CSV = "gsm8k_samples.csv"
SIMPLE_QA_CSV_PATH = DATA_DIR / SIMPLE_QA_CSV
GSM8K_CSV_PATH = DATA_DIR / GSM8K_CSV

# Results directory structure
RESULTS_ROOT = DATA_DIR / "results"
DATASET_RESULTS = {
    "gsm8k": RESULTS_ROOT / "gsm8k",
    "simple_qa": RESULTS_ROOT / "simple_qa"
}

# Output directory structure
OUTPUT_ROOT = DATA_DIR / "output"
DATASET_OUTPUT = {
    "gsm8k": OUTPUT_ROOT / "gsm8k",
    "simple_qa": OUTPUT_ROOT / "simple_qa"
}

# Evaluation directory structure
EVALUATION_ROOT = DATA_DIR / "evaluation"
DATASET_EVALUATION = {
    "gsm8k": EVALUATION_ROOT / "gsm8k",
    "simple_qa": EVALUATION_ROOT / "simple_qa"
}

# Config and data subdirectories for each dataset
def get_config_dir(dataset):
    """Get the config directory for a dataset."""
    return DATASET_OUTPUT[dataset] / "configs"

def get_data_dir(dataset):
    """Get the data directory for a dataset."""
    return DATASET_OUTPUT[dataset] / "data"