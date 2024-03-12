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
    dataset_name = "sciq"
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
        "postprocessors": ["processors.first_character"]
    }

    base_args_race = {
        "input_format": "Context: {context} Question: {question}. Answers: {choices}",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "postprocessors": ["processors.first_character"]
    }

    base_args_ai2_arc_easy = {
        "input_format": "The following are multiple choice questions (with answers) about {topic}. Question: {question} Answers: {choices} Answer:",
        "choices_field": "choices",
        "target_field": "answer",
        "choices_seperator": "\n",
        "enumerator": "numbers",
        "postprocessors": ["processors.first_character"]
    }

    dataset_name = "sciq"
    base_args = base_args_sciq
    dataset_name = "race"
    base_args = base_args_race
    dataset_name = "arc"
    base_args = base_args_ai2_arc_easy

    # Override options for different parameters
    override_options = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman"],
        "choices_seperator": [" ", "\n", ", ", "; ", " | ", "OR", "or"],
        # Add more parameters and their possible values as needed
    }

    # Create templates
    generator = MultipleChoiceTemplateGenerator(base_args, override_options)
    created_templates = generator.create_templates()

    # Save templates to local catalog
    catalog_manager = CatalogManager(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH / dataset_name)
    for i, template in enumerate(created_templates):
        catalog_manager.save_to_catalog(template, f"template_{i}")
