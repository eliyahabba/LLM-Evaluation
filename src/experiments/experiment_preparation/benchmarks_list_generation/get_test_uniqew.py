import json
import sys


def generate_pairs(data):
    """
    Generate pairs of prompts and templates according to the specified rules:
    1. All prompt paraphrases paired with the first template
    2. First prompt paraphrase paired with templates from the first group only
       (all templates sharing the same prefix up to shuffleChoices_)
    """
    results = {}

    for dataset_key, dataset_value in data.items():
        prompt_paraphrases = dataset_value["prompt_paraphrases"]
        template_names = dataset_value["template_names"]

        # Part 1: Create pairs of each prompt_paraphrase with the first template_name
        first_template = template_names[0]
        all_prompts_first_template = [(prompt, first_template) for prompt in prompt_paraphrases]

        # Part 2: For the first prompt_paraphrase, pair with all templates
        # from the first format group only
        first_prompt = prompt_paraphrases[0]

        # Get the prefix of the first template (everything up to shuffleChoices_)
        if '_shuffleChoices_' in first_template:
            first_prefix = first_template.split('_shuffleChoices_')[0]

            # Find all templates with the same prefix
            first_group_templates = []
            for template in template_names:
                if '_shuffleChoices_' in template:
                    prefix = template.split('_shuffleChoices_')[0]
                    if prefix == first_prefix:
                        first_group_templates.append(template)
        else:
            # If there's no shuffleChoices in the first template, just use it
            first_group_templates = [first_template]

        # Create pairs with the first prompt
        first_prompt_first_group_pairs = [(first_prompt, template) for template in first_group_templates]

        # Store both sets of pairs
        results[dataset_key] = {
            "all_prompts_with_first_template": all_prompts_first_template,
            "first_prompt_with_first_group_templates": first_prompt_first_group_pairs
        }

    return results


def parse_json_and_generate_pairs(json_data):
    """Parse JSON data and generate pairs"""
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    return generate_pairs(data)


def convert_tuples_to_lists(obj):
    """
    Convert tuples to lists in a nested structure, for JSON serialization.
    """
    if isinstance(obj, tuple):
        return list(convert_tuples_to_lists(item) for item in obj)
    elif isinstance(obj, list):
        return list(convert_tuples_to_lists(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_tuples_to_lists(value) for key, value in obj.items()}
    else:
        return obj


def save_pairs_to_json(pairs, output_file):
    """
    Save the pairs to a JSON file, converting tuples to lists first.
    """
    # Convert tuples to lists for JSON serialization
    serializable_pairs = convert_tuples_to_lists(pairs)

    # Create template_types dictionary (simplified version)
    template_types = {}
    for dataset_key, dataset_pairs in pairs.items():
        prompts = [pair[0] for pair in dataset_pairs["all_prompts_with_first_template"]]
        templates = [pair[1] for pair in dataset_pairs["first_prompt_with_first_group_templates"]]

        template_types[dataset_key] = {
            "prompts": prompts,
            "first_group_templates": templates
        }

    # Prepare the output data
    output_data = {
        "template_pairs_dict": serializable_pairs,
        "template_types": template_types
    }

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Template pairs saved to: {output_file}")


# Example usage:
if __name__ == "__main__":
    # Get input and output files from command line arguments or use defaults
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = r'/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/benchmarks_list_generation/experiments_config.json'

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'template_pairs_output.json'

    # When used as a script, process data from a file
    with open(input_file, 'r') as f:
        data = json.load(f)

    pairs = generate_pairs(data)

    # Save to JSON file
    save_pairs_to_json(pairs, output_file)

    # Output summary
    for dataset_key, dataset_pairs in pairs.items():
        print(f"Dataset: {dataset_key}")
        print(
            f"  - {len(dataset_pairs['all_prompts_with_first_template'])} pairs of all prompts with first template")
        print(
            f"  - {len(dataset_pairs['first_prompt_with_first_group_templates'])} pairs of first prompt with first group templates")

        # Print some examples
        print("\nExamples - All prompts with first template:")
        for pair in dataset_pairs["all_prompts_with_first_template"]:
            print(f"  {pair[0]} - {pair[1]}")

        print("\nExamples - First prompt with first group templates:")
        for pair in dataset_pairs["first_prompt_with_first_group_templates"]:
            print(f"  {pair[0]} - {pair[1]}")
        print()