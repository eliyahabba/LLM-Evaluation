import json


# To save to a file:
def save_to_json(data, filename):
    """
    Saves the dictionary to a JSON file.

    Args:
        data (dict): Dictionary to save
        filename (str): Name of the output file
    """
    path = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/conversions/hf_map_data/"
    filename = path + filename
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def create_question_metadata(questions, source):
    """
    Creates a JSON object from Hugging Face dataset questions with source and index metadata.

    Args:
        dataset_name (str): Name of the dataset to load

    Returns:
        dict: Dictionary with questions as keys and metadata as values
    """
    # Load the dataset
    # Process training set
    result = {}
    for idx, question in enumerate(questions):
        result[question] = {
            "source": source,
            "index": idx
        }
    return result


def add_ai_Arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    question = ds['test']['question']
    ds2 = load_dataset("allenai/ai2_arc", "ARC-Easy")
    question2 = ds2['test']['question']
    test = create_question_metadata(question, 'test')
    test2 = create_question_metadata(question2, 'test')
    test.update(test2)
    return test


def add_social():
    ds = load_dataset("allenai/social_i_qa")
    question = ds['train']['question']
    test = create_question_metadata(question, 'train')
    question2 = ds['validation']['question']
    test.update(create_question_metadata(question2, 'validation'))
    return test


def add_openbook():
    ds = load_dataset("allenai/openbookqa")
    question = ds['test']['question_stem']
    test = create_question_metadata(question, 'test')
    question2 = ds['train']['question_stem']
    test.update(create_question_metadata(question2, 'train'))
    return test


def get_hell():
    ds = load_dataset("Rowan/hellaswag")
    question = ds['test']['ctx']
    test = create_question_metadata(question, 'test')
    question2 = ds['validation']['ctx']
    test.update(create_question_metadata(question2, 'validation'))
    return test


def get_mmlu():
    ds = load_dataset(f"cais/mmlu", "all")
    question = ds['test']['question']
    test = create_question_metadata(question, 'test')
    return test

def get_mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    question = ds['test']['question']
    test = create_question_metadata(question, 'test')
    return test


if __name__ == '__main__':
    from datasets import load_dataset

    # enumerated_questions = add_ai_Arc()
    # save_to_json(enumerated_questions, 'ai2_arc_samples.json')
    # #
    # enumerated_questions = add_social()
    # save_to_json(enumerated_questions, 'social_iqa_samples.json')
    # #
    # questions = add_openbook()
    # save_to_json(questions, 'openbook_qa_samples.json')
    #
    # questions = get_hell()
    # save_to_json(questions, 'hellaswag_samples.json')

    questions = get_mmlu()
    save_to_json(questions, 'mmlu_samples.json')

    questions = get_mmlu_pro()
    save_to_json(questions, 'mmlu_pro_samples.json')
