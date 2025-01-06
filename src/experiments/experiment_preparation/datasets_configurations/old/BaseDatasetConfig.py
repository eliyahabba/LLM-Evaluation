from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams


class BaseDatasetConfig:
    def __init__(self, kwargs=None):
        self.choices_field = "choices" if kwargs is None or "choices_field" not in kwargs else kwargs["choices_field"]
        self.target_field = "answer" if kwargs is None or "target_field" not in kwargs else kwargs["target_field"]
        self.choices_separator = "\n" if kwargs is None or "choices_separator" not in kwargs else kwargs[
            "choices_separator"]
        self.enumerator = "numbers" if kwargs is None or "enumerator" not in kwargs else kwargs["enumerator"]
        self.source_choice_format = "{choice_numeral}. {choice_text}"
        self.target_choice_format = "{choice_numeral}. {choice_text}"
        self.postprocessors = [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
        self._get_input_format(kwargs["input_format_func"])
        self._slice_shuffled_choices_arguments(kwargs[
                                                   "shuffle_choices"])

    def to_dict(self):
        return {
            "input_format": self.input_format,
            "choices_field": self.choices_field,
            "target_field": self.target_field,
            "choices_separator": self.choices_separator,
            "enumerator": self.enumerator,
            "source_choice_format": self.source_choice_format,
            "target_choice_format": self.target_choice_format,
            "postprocessors": self.postprocessors,
            "shuffle_choices": self.shuffle_choices,
            "sort_choices_by_length": self.sort_choices_by_length,
            "sort_choices_alphabetically": self.sort_choices_alphabetically,
            "reverse_choices": self.reverse_choices
        }

    def _get_input_format(self, input_format_func_name):
        input_format_func = getattr(self, input_format_func_name)
        self.input_format = input_format_func().text

    def _slice_shuffled_choices_arguments(self, shuffle_choices: str) -> None:
        shuffle_choices_dict = ConfigParams.get_shuffle_choices_argument(shuffle_choices)
        self.shuffle_choices = shuffle_choices_dict["shuffle_choices"]
        self.sort_choices_by_length = shuffle_choices_dict["sort_choices_by_length"]
        self.sort_choices_alphabetically = shuffle_choices_dict["sort_choices_alphabetically"]
        self.reverse_choices = shuffle_choices_dict["reverse_choices"]
