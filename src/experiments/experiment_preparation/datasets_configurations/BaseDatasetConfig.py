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

    def get_mmlu_instructions_with_topic(self, topic: str) -> str:
        return f"The following are multiple choice questions (with answers) about {topic}.\n"

    def get_mmlu_instructions_without_topic(self) -> str:
        return "The following are multiple choice questions (with answers).\n"

    def get_structured_instruction_text(self, context_topic):
        # Provide a default implementation if needed
        return f"{context_topic}Question: {{question}} Choices: {{choices}} Answer: {{answer}}\n"

    def get_context_topic(self):
        # Base class provides a generic placeholder which can be overridden
        return ""

    def get_input_format(self):
        raise NotImplementedError("Subclasses must implement this method to return the specific input format.")
