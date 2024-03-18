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
        "input_format": "Context: {context} Question: {question} Choices: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "postprocessors": ["processors.first_character"]
    }

    base_args_race = {
        "input_format": "Context: {context} Question: {question}. Answers: {choices}. Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "postprocessors": ["processors.first_character"]
    }

    base_args_ai2_arc_easy = {
        "input_format": "The following are multiple choice questions (with answers) about {topic}. Question: {question} Answers: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "source_choice_format": "{choice_numeral}. {choice_text}",
        "target_choice_format": "{choice_numeral}",
        "postprocessors": ["processors.first_character"]
    }
    datasets_templates = [base_args_sciq, base_args_race, base_args_ai2_arc_easy]
    dataset_names = ["sciq", "race_all", "ai2_arc.arc_easy"]
    dataset_names_to_templates = dict(zip(dataset_names, datasets_templates))
    override_options = {
        "target_choice_format": ["{choice_numeral}", "{choice_numeral}. {choice_text}", "{choice_text}"],
        "enumerator": ["capitals", "lowercase", "numbers", "roman"],
        "choices_seperator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
        # Add more parameters and their possible values as needed
    }
