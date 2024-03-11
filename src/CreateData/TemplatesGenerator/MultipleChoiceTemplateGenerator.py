from unitxt.templates import MultipleChoiceTemplate

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.TemplatesGenerator.TemplateGenerator import TemplateGenerator
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class MultipleChoiceTemplateGenerator(TemplateGenerator):
    def create_template(self, **override_args) -> MultipleChoiceTemplate:
        """
        Creates a MultipleChoiceTemplate instance with specific parameters.

        @param: base_args (dict): A dictionary containing base arguments for the template.
        @param override_args: A dictionary containing override options for the template.

        @return: MultipleChoiceTemplate: The created template instance.
        """
        args = {**self.base_args, **override_args}
        template = MultipleChoiceTemplate(**args)
        return template


if __name__ == "__main__":
    # Base arguments for all templates
    base_args = {
        "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion:"
                        " {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:",
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
        # Add more parameters and their possible values as needed
    }

    # Create templates
    generator = MultipleChoiceTemplateGenerator(base_args, override_options)
    created_templates = generator.create_templates()

    # Save templates to local catalog
    catalog_manager = CatalogManager(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH)
    for i, template in enumerate(created_templates):
        catalog_manager.save_to_catalog(template, f"template_{i}")
