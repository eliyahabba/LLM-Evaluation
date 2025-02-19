import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


def save_to_csv(data, filename, num_of_samples=None):
    """
    Saves the data to a CSV file.

    Args:
        data (dict): Dictionary with metadata
        filename (str): Name of the output file
    """
    path = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/conversions/hf_map_data/"
    filename = path + filename

    # Convert the data to a format suitable for DataFrame
    records = []

    for question, choices in data.keys():
        metadata = data[(question, choices)]
        record = {
            'question': question,
            'choices': choices,  # Store choices directly as they are
            'source': metadata['source'],
            'index': metadata['index']
        }
        records.append(record)
    if num_of_samples:
        records = records[:num_of_samples]
    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    # df.to_csv(filename, index=False, encoding='utf-8')
    df.to_parquet(filename, index=False)


def create_question_metadata(questions, choices_list, source,dataname):
    """
    Creates metadata dictionary for questions and their choices.

    Args:
        questions (list): List of questions
        choices_list (list): List of answer choices for each question
        source (str): Source of the dataset

    Returns:
        dict: Dictionary with (question, choices) tuples as keys and metadata as values
    """
    result = {}
    for idx, (question, choices) in enumerate(zip(questions, choices_list)):
        # Store choices directly as a string if it's already a string,
        # otherwise keep it as is
        if (question, choices if isinstance(choices, str) else tuple(choices)) in result:
            print(f"Duplicate question: {dataname}: {source}: {idx}")
            continue
        result[(question, choices if isinstance(choices, str) else tuple(choices))] = {
            "source": source,
            "index": idx
        }
    return result


def add_ai_Arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    questions = ds['test']['question']
    choices = [example['choices']['text'] for example in ds['test']]
    test = create_question_metadata(questions, choices, 'test', dataname = 'ARC-Challenge')

    ds2 = load_dataset("allenai/ai2_arc", "ARC-Easy")
    questions2 = ds2['test']['question']
    choices2 = [example['choices']['text'] for example in ds2['test']]
    test2 = create_question_metadata(questions2, choices2, 'test', dataname = 'ARC-Easy')
    test.update(test2)
    return test


def add_social():
    ds = load_dataset("allenai/social_i_qa")
    questions = ds['train']['question']
    choices = [[ex['answerA'], ex['answerB'], ex['answerC']] for ex in ds['train']]
    test = create_question_metadata(questions, choices, 'train', dataname = 'social_i_qa')

    questions2 = ds['validation']['question']
    choices2 = [[ex['answerA'], ex['answerB'], ex['answerC']] for ex in ds['validation']]
    test.update(create_question_metadata(questions2, choices2, 'validation', dataname = 'social_i_qa'))
    return test


def add_openbook():
    ds = load_dataset("allenai/openbookqa")
    questions = ds['test']['question_stem']
    choices = [example['choices']['text'] for example in ds['test']]
    test = create_question_metadata(questions, choices, 'test', dataname = 'openbookqa')

    questions2 = ds['train']['question_stem']
    choices2 = [example['choices']['text'] for example in ds['train']]
    test.update(create_question_metadata(questions2, choices2, 'train', dataname = 'openbookqa'))
    return test


def get_hell():
    ds = load_dataset("Rowan/hellaswag")
    questions = ds['test']['ctx']
    choices = [example['endings'] for example in ds['test']]
    test = create_question_metadata(questions, choices, 'test', dataname = 'hellaswag')

    questions2 = ds['validation']['ctx']
    choices2 = [example['endings'] for example in ds['validation']]
    test.update(create_question_metadata(questions2, choices2, 'validation', dataname = 'hellaswag'))
    return test


def get_mmlu():
    ds = load_dataset("cais/mmlu", "all")
    questions = ds['test']['question']
    choices = ds['test']['choices']
    test = create_question_metadata(questions, choices, 'test', dataname = 'mmlu')
    return test


def get_mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    questions = ds['test']['question']
    choices = ds['test']['options']
    test = create_question_metadata(questions, choices, 'test', dataname = 'mmlu_pro')
    return test


def get_race_middle():
    ds = load_dataset("ehovy/race", "middle", split='test')
    questions = ds['question']
    choices = ds['options']
    test = create_question_metadata(questions, choices, 'test', dataname = 'race_middle')
    return test


def get_race_high():
    ds = load_dataset("ehovy/race", "high", split='test')
    questions = ds['question']
    choices = ds['options']
    test = create_question_metadata(questions, choices, 'test', dataname = 'race_high')
    return test


def get_global_mmlu(lang):
    ds = load_dataset("CohereForAI/Global-MMLU", lang, split='test')
    questions = ds['question']
    # Combine the separate option columns into a list for each question
    choices = [
        [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
        for row in ds
    ]
    test = create_question_metadata(questions, choices, 'test', dataname = 'global_mmlu')
    return test
def get_global_mmlu_ca(lang):
    ds = load_dataset("CohereForAI/Global-MMLU-Lite", lang, split='test')
    # filter by cultural_sensitivity_label ==CA
    ds = ds.filter(lambda x: x['cultural_sensitivity_label'] == 'CA')
    questions = ds['question']
    # Combine the separate option columns into a list for each question
    choices = [
        [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
        for row in ds
    ]
    test = create_question_metadata(questions, choices, 'test', dataname = 'global_mmlu_lite_ca')
    return test

def get_global_mmlu_cs(lang):
    ds = load_dataset("CohereForAI/Global-MMLU-Lite", lang, split='test')
    # filter by cultural_sensitivity_label ==CS
    ds = ds.filter(lambda x: x['cultural_sensitivity_label'] == 'CS')
    questions = ds['question']
    # Combine the separate option columns into a list for each question
    choices = [
        [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
        for row in ds
    ]
    test = create_question_metadata(questions, choices, 'test', dataname = 'global_mmlu_lite_cs')
    return test
if __name__ == '__main__':
    # Run all the dataset processing functions
    enumerated_questions = add_ai_Arc()
    save_to_csv(enumerated_questions, 'ai2_arc_samples.parquet')

    enumerated_questions = add_social()
    save_to_csv(enumerated_questions, 'social_iqa_samples.parquet', num_of_samples=200)

    questions = add_openbook()
    save_to_csv(questions, 'openbook_qa_samples.parquet', num_of_samples=200)

    questions = get_hell()
    save_to_csv(questions, 'hellaswag_samples.parquet', num_of_samples=200)

    questions = get_mmlu()
    save_to_csv(questions, 'mmlu_samples.parquet')

    questions = get_mmlu_pro()
    save_to_csv(questions, 'mmlu_pro_samples.parquet')

    questions = get_race_middle()
    save_to_csv(questions, 'race_middle_samples.parquet', num_of_samples=200)

    questions = get_race_high()
    save_to_csv(questions, 'race_high_samples.parquet', num_of_samples=200)

    SUPPORTED_LANGUAGES = {
        "ar": "Arabic", "bn": "Bengali", "de": "German",
        "fr": "French", "hi": "Hindi", "id": "Indonesian",
        "it": "Italian", "ja": "Japanese", "ko": "Korean",
        "pt": "Portuguese", "es": "Spanish", "sw": "Swahili",
        "yo": "Yorùbá", "zh": "Chinese"
    }

    for lang in tqdm(SUPPORTED_LANGUAGES.keys()):
        questions = get_global_mmlu(lang)
        save_to_csv(questions, f'global_mmlu.{lang}_samples.parquet', num_of_samples=200)
        questions = get_global_mmlu_ca(lang)
        save_to_csv(questions, f'global_mmlu_lite_ca.{lang}_samples.parquet')
        questions = get_global_mmlu_cs(lang)
        save_to_csv(questions, f'global_mmlu_lite_cs.{lang}_samples.parquet')
