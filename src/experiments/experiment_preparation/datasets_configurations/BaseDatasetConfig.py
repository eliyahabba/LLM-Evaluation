class BaseDatasetConfig:
    def __init__(self, kwargs=None):
        self.choices_field = "choices" if kwargs is None or "choices_field" not in kwargs else kwargs["choices_field"]
        self.target_field = "answer" if kwargs is None or "target_field" not in kwargs else kwargs["target_field"]
        self.choices_separator = "\n" if kwargs is None or "choices_separator" not in kwargs else kwargs[
            "choices_separator"]
        self.enumerator = "numbers" if kwargs is None or "enumerator" not in kwargs else kwargs["enumerator"]
        self.source_choice_format = "{choice_numeral}. {choice_text}"
        self.target_choice_format = "{choice_numeral}. {choice_text}"
        self.shuffle_choices = False if kwargs is None or "shuffle_choices" not in kwargs else kwargs["shuffle_choices"]
        self.postprocessors = [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
        input_format_func_name = kwargs["input_format_func"]
        input_format_func = getattr(self, input_format_func_name)
        self.input_format = input_format_func()

    def to_dict(self):
        return {
            "input_format": self.input_format,
            "choices_field": self.choices_field,
            "target_field": self.target_field,
            "choices_separator": self.choices_separator,
            "enumerator": self.enumerator,
            "source_choice_format": self.source_choice_format,
            "target_choice_format": self.target_choice_format,
            "shuffle_choices": self.shuffle_choices,
            "postprocessors": self.postprocessors
        }
