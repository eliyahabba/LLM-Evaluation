class BaseDatasetConfig:
    def __init__(self):
        self.choices_field = "choices"
        self.target_field = "answer"
        self.choices_separator = "\n"
        self.enumerator = "numbers"
        self.source_choice_format = "{choice_numeral}. {choice_text}"
        self.target_choice_format = "{choice_numeral}. {choice_text}"
        self.shuffle_choices = False
        self.postprocessors = [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]

    def to_dict(self):
        return {
            "input_format": self.get_input_format(),
            "choices_field": self.choices_field,
            "target_field": self.target_field,
            "choices_separator": self.choices_separator,
            "enumerator": self.enumerator,
            "source_choice_format": self.source_choice_format,
            "target_choice_format": self.target_choice_format,
            "shuffle_choices": self.shuffle_choices,
            "postprocessors": self.postprocessors
        }

    def get_input_format(self):
        raise NotImplementedError("Subclasses must implement this method to return the specific input format.")
