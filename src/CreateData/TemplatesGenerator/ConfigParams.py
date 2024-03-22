class ConfigParams:
    base_args_copa = {
        "input_format": "The following are multiple choice questions (with answers)\n\nQuestion:"
                        " {question}\nChoose the correct answer from {numerals}\nAnswers:\n{choices}\nAnswer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "postprocessors": ["processors.first_character"]
    }

    base_args_sciq = {
        "input_format": "Context: [context] Question: [question] Choices: [choices] Answer: [answer]\nContext: {context} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "shuffle_choices": False,
        "postprocessors": ["processors.first_character"]
    }

    base_args_race = {
        "input_format": "Context: [context] Question: [question] Choices: [choices] Answer: [answer]\nContext: {context} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "shuffle_choices": False,
        "postprocessors": ["processors.first_character"]
    }

    base_args_ai2_arc_easy = {
        "input_format": "Context: [context] Question: [question] Choices: [choices] Answer: [answer]\nContext: {context} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "shuffle_choices": False,
        "postprocessors": ["processors.first_character"]
    }
    base_args_mmlu_global_facts = {
        "input_format": "Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "shuffle_choices": False,
        "postprocessors": ["processors.first_character"]
    }

    base_args_mmlu_machine_learning = {
        "input_format": "Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "shuffle_choices": False,
        "postprocessors": ["processors.first_character"]
    }

    datasets_templates = [base_args_sciq, base_args_race, base_args_ai2_arc_easy, base_args_mmlu_global_facts,
                          base_args_mmlu_machine_learning]
    dataset_names = ["sciq", "race_all", "ai2_arc.arc_easy", "mmlu.global_facts", "mmlu.machine_learning"]
    dataset_names_to_templates = dict(zip(dataset_names, datasets_templates))
    override_options = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman"],
        "choices_seperator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
        "shuffle_choices": [False, True],
        # Add more parameters and their possible values as needed
    }
    map_enumerator = {"ABCDEFGHIJKLMNOP": "capitals",
                      "abcdefghijklmnop": "lowercase",
                      str(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19', '20']): "numbers",
                      str(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
                          'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']): "roman"}
