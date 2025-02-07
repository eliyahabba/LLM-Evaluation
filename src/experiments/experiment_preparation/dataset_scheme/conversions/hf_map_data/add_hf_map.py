import json

from tqdm import tqdm


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
        json.dump(data, f, indent=4, ensure_ascii=False)


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

def get_global_mmlu(lang):
    ds = load_dataset("CohereForAI/Global-MMLU", lang, split='test')
    questions = ds['question']
    test = create_question_metadata(questions, 'test')
    return test


def get_global_mmlu_ca(lang):
    ds = load_dataset("CohereForAI/Global-MMLU-Lite", lang, split='test')
    # filet by cultural_sensitivity_label ==CA
    ds = ds.filter(lambda x: x['cultural_sensitivity_label'] == 'CA')
    questions = ds['question']
    test = create_question_metadata(questions, 'test')
    return test


def get_global_mmlu_cs(lang):
    ds = load_dataset("CohereForAI/Global-MMLU-Lite", lang, split='test')
    # filet by cultural_sensitivity_label ==CA
    ds = ds.filter(lambda x: x['cultural_sensitivity_label'] == 'CS')
    questions = ds['question']
    test = create_question_metadata(questions, 'test')
    return test


def get_race_middle():
    ds = load_dataset("ehovy/race", "middle", split='test')
    questions = ds['question']
    test = create_question_metadata(questions, 'test')
    return test

def get_race_high():
    ds = load_dataset("ehovy/race", "high", split='test')
    questions = ds['question']
    test = create_question_metadata(questions, 'test')
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

    # questions = get_mmlu()
    # save_to_json(questions, 'mmlu_samples.json')
    #
    # questions = get_mmlu_pro()
    # save_to_json(questions, 'mmlu_pro_samples.json')

    questions = get_race_middle()
    save_to_json(questions, 'race_middle_samples.json')

    questions = get_race_high()
    save_to_json(questions, 'race_high_samples.json')

    SUPPORTED_LANGUAGES = {
        "ar": "Arabic",
        "bn": "Bengali",
        "de": "German",
        "fr": "French",
        "hi": "Hindi",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "es": "Spanish",
        "sw": "Swahili",
        "yo": "Yorùbá",
        "zh": "Chinese"
    }
    # for lang in tqdm(SUPPORTED_LANGUAGES.keys()):
    #     questions = get_global_mmlu(lang)
    #     save_to_json(questions, f'global_mmlu.{lang}_samples.json')
        #
        # questions = get_global_mmlu_ca(lang)
        # save_to_json(questions, f'global_mmlu_lite_ca.{lang}_samples.json')
        # # #
        # #
        # questions = get_global_mmlu_cs(lang)
        # save_to_json(questions, f'global_mmlu_lite_cs.{lang}_samples.json')
