from itertools import product
from unitxt.templates import MultipleChoiceTemplate
from CatalogManager import CatalogManager

from src.utils.Constants import Constants

UnitxtDataConstants = Constants.UnitxtDataConstants


class TemplateGenerator:
    def __init__(self, base_args: dict, override_options: dict):
        """
        Initializes the TemplateGenerator with base arguments and override options.

        @param base_args: A dictionary containing base arguments for the template.
        @param override_options: A dictionary containing override options for the template.
        """
        self.base_args = base_args
        self.override_options = override_options

    def create_templates(self) -> list:
        """
        Creates a list of MultipleChoiceTemplate instances with different parameters.

        @return: A list of the created templates.
        """
        templates = []
        for options in product(*self.override_options.values()):
            override_args = dict(zip(self.override_options.keys(), options))
            args = {**self.base_args, **override_args}
            template = MultipleChoiceTemplate(**args)
            templates.append(template)
            print(f"Created template with options: {override_args}")
        print("All templates created!")
        return templates

    @staticmethod
    def create_template(base_args: dict, **override_args) -> MultipleChoiceTemplate:
        """
        Creates a MultipleChoiceTemplate instance with specific parameters.

        @param: base_args (dict): A dictionary containing base arguments for the template.
        @param override_args: A dictionary containing override options for the template.

        @return: MultipleChoiceTemplate: The created template instance.
        """
        args = {**base_args, **override_args}
        template = MultipleChoiceTemplate(**args)
        return template


if __name__ == "__main__":
    # Base arguments for all templates
    base_args = {
        "input_format": "What is the capital of {topic}?\n\nChoose from:\n{choices}",
        "choices_field": "options",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": ["numbers", "capitals", "lowercase"],
        "source_choice_format": ["{choice_text}", "{choice_numeral}. {choice_text}"],
        "target_choice_format": ["{choice_numeral}", "{choice_text}"],
    }

    # Override options for different parameters
    override_options = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman"],
        "target_field": ["answer", "correct_answer", "response"],
        # Add more parameters and their possible values as needed
    }

    # Create templates
    generator = TemplateGenerator(base_args, override_options)
    created_templates = generator.create_templates()

    # Save templates to local catalog
    saver = CatalogManager(UnitxtDataConstants.CATALOG_PATH)
    for i, template in enumerate(created_templates):
        saver.save_to_catalog(template, f"template_{i}")
