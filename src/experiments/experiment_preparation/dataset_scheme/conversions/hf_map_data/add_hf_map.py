from datasets import load_dataset

import json



# To save to a file:
def save_to_json(data, filename):
    """
    Saves the dictionary to a JSON file.

    Args:
        data (dict): Dictionary to save
        filename (str): Name of the output file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)



def create_numbered_json(string_list):
    """
    Creates a JSON object from a list of strings where each string is a key
    and its index is the value.

    Args:
        string_list (list): List of strings to convert

    Returns:
        dict: Dictionary with strings as keys and their indices as values
    """
    # Create dictionary comprehension with enumerated items
    result = {item: idx for idx, item in enumerate(string_list)}
    return result

def add_ai_Arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    question = ds['test']['question']
    ds2 = load_dataset("allenai/ai2_arc", "ARC-Easy")
    question2 = ds2['test']['question']
    question = create_numbered_json(question)
    question2 = create_numbered_json(question2)
    question.update(question2)
    return question

def add_social():
    ds = load_dataset("allenai/social_i_qa")
    question = ds['train']['question']
    question = create_numbered_json(question)
    return question

def add_openbook():
    ds = load_dataset("allenai/openbookqa")
    question = ds['test']['question_stem']

    return question
def get_hell():
    ds = load_dataset("Rowan/hellaswag")
    question = ds['test']['ctx']
    return question
from datasets import load_dataset

enumerated_questions = add_ai_Arc()
save_to_json(enumerated_questions, 'ai2_arc_samples.json')

enumerated_questions = add_social()
save_to_json(enumerated_questions, 'social_iqa_samples.json')

questions = add_openbook()
save_to_json(create_numbered_json(questions), 'openbook_qa_samples.json')

questions = get_hell()
save_to_json(create_numbered_json(questions), 'hellaswag_samples.json')