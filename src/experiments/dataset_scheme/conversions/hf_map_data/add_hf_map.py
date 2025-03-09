import json

from datasets import load_dataset
from tqdm import tqdm


def save_to_csv(data, filename, num_of_samples=None):
    """Saves the data to both JSON (for fast lookup) and CSV/Parquet (for compatibility)."""
    path = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/conversions/hf_map_data/"
    base_filename = path + filename.replace('.parquet', '').replace('.csv', '')

    # Save as JSON for fast lookup - no need for conversion since keys are strings
    with open(f"{base_filename}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    #
    # # Convert to records for DataFrame
    # records = []
    # for key, metadata in data.items():
    #     question, *choices = key.split("|||")
    #     record = {
    #         'question': question,
    #         'choices': metadata['choices'],  # Use original order
    #         'source': metadata['source'],
    #         'index': metadata['index']
    #     }
    #     records.append(record)
    #
    # # Apply sample limit if specified
    # if num_of_samples:
    #     records = records[:num_of_samples]
    #
    # # Save as DataFrame formats
    # df = pd.DataFrame(records)
    # df.to_parquet(f"{base_filename}.parquet", index=False)
    # df.to_csv(f"{base_filename}.csv", index=False, encoding='utf-8')


def create_question_metadata(questions, choices_list, source, dataname):
    """
    Creates metadata dictionary for questions and their choices with efficient lookup.
    
    Returns:
        dict: Dictionary with question+sorted_choices as key for O(1) lookup:
        {
            "question|||choice1,choice2,choice3": {"index": idx, "source": source, "choices": [original choices]}
        }
    """
    result = {}
    for idx, (question, choices) in enumerate(zip(questions, choices_list)):
        # Normalize choices to list format
        choices_list = choices if isinstance(choices, list) else choices.split(", ")

        # Create a unique key by combining question with sorted choices
        key = f"{question}|||{'|||'.join(sorted(choices_list))}"

        # Store metadata
        result[key] = {
            "source": source,
            "index": idx,
            "choices": choices_list  # Keep original order
        }

    return result


def add_ai_Arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    questions = ds['test']['question']
    choices = [example['choices']['text'] for example in ds['test']]
    test = create_question_metadata(questions, choices, 'test', dataname='ARC-Challenge')

    ds2 = load_dataset("allenai/ai2_arc", "ARC-Easy")
    questions2 = ds2['test']['question']
    choices2 = [example['choices']['text'] for example in ds2['test']]
    test2 = create_question_metadata(questions2, choices2, 'test', dataname='ARC-Easy')
    test.update(test2)
    return test


def add_social():
    ds = load_dataset("allenai/social_i_qa")
    questions = ds['train']['question']
    choices = [[ex['answerA'], ex['answerB'], ex['answerC']] for ex in ds['train']]
    test = create_question_metadata(questions, choices, 'train', dataname='social_i_qa')

    questions2 = ds['validation']['question']
    choices2 = [[ex['answerA'], ex['answerB'], ex['answerC']] for ex in ds['validation']]
    test.update(create_question_metadata(questions2, choices2, 'validation', dataname='social_i_qa'))
    return test


def add_openbook():
    ds = load_dataset("allenai/openbookqa")
    questions = ds['test']['question_stem']
    choices = [example['choices']['text'] for example in ds['test']]
    test = create_question_metadata(questions, choices, 'test', dataname='openbookqa')

    questions2 = ds['train']['question_stem']
    choices2 = [example['choices']['text'] for example in ds['train']]
    test.update(create_question_metadata(questions2, choices2, 'train', dataname='openbookqa'))
    return test


def get_hell():
    ds = load_dataset("Rowan/hellaswag")
    questions = ds['test']['ctx']
    choices = [example['endings'] for example in ds['test']]
    test = create_question_metadata(questions, choices, 'test', dataname='hellaswag')

    questions2 = ds['validation']['ctx']
    choices2 = [example['endings'] for example in ds['validation']]
    test.update(create_question_metadata(questions2, choices2, 'validation', dataname='hellaswag'))
    return test


def get_mmlu():
    ds = load_dataset("cais/mmlu", "all")
    questions = ds['test']['question']
    choices = ds['test']['choices']
    test = create_question_metadata(questions, choices, 'test', dataname='mmlu')
    return test


def get_mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    questions = ds['test']['question']
    choices = ds['test']['options']
    test = create_question_metadata(questions, choices, 'test', dataname='mmlu_pro')
    return test


def get_race_middle():
    ds = load_dataset("ehovy/race", "middle", split='test')
    questions = ds['question']
    choices = ds['options']
    test = create_question_metadata(questions, choices, 'test', dataname='race_middle')
    return test


def get_race_high():
    ds = load_dataset("ehovy/race", "high", split='test')
    questions = ds['question']
    choices = ds['options']
    test = create_question_metadata(questions, choices, 'test', dataname='race_high')
    return test


def get_global_mmlu(lang):
    ds = load_dataset("CohereForAI/Global-MMLU", lang, split='test')
    questions = ds['question']
    # Combine the separate option columns into a list for each question
    choices = [
        [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
        for row in ds
    ]
    test = create_question_metadata(questions, choices, 'test', dataname='global_mmlu')
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
    test = create_question_metadata(questions, choices, 'test', dataname='global_mmlu_lite_ca')
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
    test = create_question_metadata(questions, choices, 'test', dataname='global_mmlu_lite_cs')
    return test


def get_quality():
    """Load and process the Quilty dataset."""
    ds = load_dataset("emozilla/quality")

    # Get train data
    questions = ds['train']['question']
    choices = ds['train']['options']
    test = create_question_metadata(questions, choices, 'train', dataname='quality')

    # Get validation data and update the dictionary
    questions_val = ds['validation']['question']
    choices_val = ds['validation']['options']
    test.update(create_question_metadata(questions_val, choices_val, 'validation', dataname='quality'))

    return test


if __name__ == '__main__':
    # Run all the dataset processing functions
    enumerated_questions = add_ai_Arc()
    save_to_csv(enumerated_questions, 'ai2_arc_samples.parquet')

    enumerated_questions = add_social()
    save_to_csv(enumerated_questions, 'social_iqa_samples.parquet')

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

    # Add Quilty dataset processing before the language loop
    questions = get_quality()
    save_to_csv(questions, 'quality_samples.parquet')

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
