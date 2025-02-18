from datasets import load_dataset
from difflib import get_close_matches
import sys

from tqdm import tqdm

HUGGINGFACE_DATA_PATH = "eliyahabba/llm-evaluation-without-probs"
FILE_EXAMPLE = "data_2024-12-23T20%3A00%3A00%2B00%3A00_2024-12-23T22%3A00%3A00%2B00%3A00_test.parquet"


def show_row_data(row_number, first_row):
    try:
        instance = first_row['instance']
        output = first_row['output']
        evaluation = first_row['evaluation']

        choices = instance['classification_fields']['choices']
        choices_text = [choice['text'] for choice in choices]
        choices_ids = [choice['id'] for choice in choices]
        model_response = output['response']
        ground_truth_text = instance['classification_fields']['ground_truth']['text']
        ground_truth_id = instance['classification_fields']['ground_truth']['id']
        ground_truth = ground_truth_id + ". " + ground_truth_text
        score = evaluation['score']

        ai_answer = model_response
        choices_text_labels = [". ".join(x) for x in list(zip(choices_ids, choices_text))]
        closest_answer = get_closest_answer(ai_answer, choices_text_labels)

        # closest_letter = choices_ids[choices_text.index(closest_answer)]

        if(ground_truth == closest_answer and score != 1 ):
            print(f"{row_number} - Ground Truth: {ground_truth}")
            print(f"{row_number} - Closest: {closest_answer}")
            print(f"{row_number} - Score: {score}\n")
            print(f"{row_number} - Response: {get_first_row(ai_answer)}")
            print("\n\n")


    except Exception as e:
        print(f"Error processing row {row_number}: {e}")
def get_first_row(text):
    answer = text.strip().split("\n")
    answer = answer[0].strip()
    return answer
def get_closest_answer(answer, options):
    ai_answer = answer.strip().split("\n")
    ai_answer = ai_answer[0].strip()
    return get_close_matches(ai_answer, options, n=1, cutoff=0.0)[0]

if __name__ == '__main__':
    try:

        # dataset = load_dataset(HUGGINGFACE_DATA_PATH, data_files=FILE_EXAMPLE, streaming=True)
        dataset = load_dataset(HUGGINGFACE_DATA_PATH, data_files=FILE_EXAMPLE)
        print(f"Dataset loaded: {dataset} with size {len(dataset)}")
        for i, first_row in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
            show_row_data(i + 1, first_row)
        exit()


    except Exception as e:
        print(f"Error loading the dataset: {e}")




"""
evaluation_id:
  e6ba5f2ca9ed0ecfa6e8fb38c9d1997ea9db039f4b3b273a5cc784cf08072189
model:
{
  "model_info": {
    "name": "meta-llama/Llama-3.2-3B-Instruct", 
    "family": "Llama"
  },
  "configuration": {
    "architecture": "transformer",
    "parameters": 3000000000,
    "context_window": 131072,
    "is_instruct": true,
    "hf_path": "meta-llama/Llama-3.2-3B-Instruct",
    "revision": null
  },
  "inference_settings": {
    "quantization": {
      "bit_precision": "float16",
      "method": "None"
    },
    "generation_args": {
      "use_vllm": true,
      "temperature": null,
      "top_p": null,
      "top_k": -1,
      "max_tokens": 64,
      "stop_sequences": []
    }
  }
}
prompt_config:
{
  "prompt_class": "MultipleChoice",
  "format": {
    "type": "MultipleChoice",
    "template": "The following are multiple choice questions (with answers).\n\nQuestion: {question}\n\n{choices}\nAnswer:",
    "separator": " OR ",
    "enumerator": "greek",
    "choices_order": {
      "method": "random",
      "description": "Randomly shuffles all choices"
    },
    "shots": 0,
    "demos": []
  }
}
instance:
{
  "task_type": "classification",
  "raw_input": [
    {
      "role": "user",
      "content": "The following are multiple choice questions (with answers).\n\nQuestion: Mass-society theory suggests that:\n\n\u03b1. audiences make selective interpretations of media messages OR \u03b2. the media manipulate 'the masses' as vulnerable, passive consumers OR \u03b3. the content of the media is determined by market forces OR \u03b4. the subordinate classes are dominated by the ideology of the ruling class\nAnswer:"
    }
  ],
  "sample_identifier": {
    "dataset_name": "mmlu.sociology",
    "hf_repo": "cais/mmlu",
    "hf_split": "test,
    "hf_index": 13404
  },
  "perplexity": 74.18370247478661,
  "classification_fields": {
    "question": "Mass-society theory suggests that:",
    "choices": [
      {
        "id": "\u03b1",
        "text": "audiences make selective interpretations of media messages"
      },
      {
        "id": "\u03b2",
        "text": "the media manipulate 'the masses' as vulnerable, passive consumers"
      },
      {
        "id": "\u03b3",
        "text": "the content of the media is determined by market forces"
      },
      {
        "id": "\u03b4",
        "text": "the subordinate classes are dominated by the ideology of the ruling class"
      }
    ],
    "ground_truth": {
      "id": "\u03b2",
      "text": "the media manipulate 'the masses' as vulnerable, passive consumers"
    }
  },
  "prompt_logprobs": []
}
output:
{
  "response": "The correct answer is: \u03b1. audiences make selective interpretations of media messages.\n\nThe Mass-Society Theory, also known as the Mass Culture Approach, suggests that audiences construct their own meanings and interpretations from media messages, rather than the media controlling or manipulating them. This theory proposes that audiences actively engage with and interpret media content in",
  "cumulative_logprob": -38.74751878141933,
  "generated_tokens_ids": [
    791,
    4495,
    4320,
    374,
    25,
    19581,
    13,
    30994,
    1304,
    44010,
    58689,
    315,
    3772,
    6743,
    382,
    791,
    9346,
    6354,
    6180,
    31535,
    11,
    1101,
    3967,
    439,
    279,
    9346,
    21023,
    54184,
    11,
    13533,
    430,
    30994,
    9429,
    872,
    1866,
    50800,
    323,
    58689,
    505,
    3772,
    6743,
    11,
    4856,
    1109,
    279,
    3772,
    26991,
    477,
    63614,
    1124,
    13,
    1115,
    10334,
    55725,
    430,
    30994,
    22815,
    16988,
    449,
    323,
    14532,
    3772,
    2262,
    304
  ],
  "generated_tokens_logprobs": [],
  "max_prob": null
}
evaluation:
{
  "evaluation_method": {
    "method_name": "content_similarity",
    "description": "Finds the most similar answer among the given choices by comparing the textual content"
  },
  "ground_truth": "\u03b2. the media manipulate 'the masses' as vulnerable, passive consumers",
  "score": 0.0
}
(.venv) niso@lake-31:/cs/labs/gabis/niso/IBMData/pythonProject$ 
"""