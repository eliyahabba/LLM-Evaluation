from tqdm import tqdm
from unitxt.templates import MultipleChoiceTemplate

from src.CreateData.CatalogManager import CatalogManager
from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
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
        if args['enumerator'] == 'roman':
            args['postprocessors'] = [
                "processors.to_string_stripped",
                "processors.take_first_non_empty_line",
                "processors.match_closest_option"
            ]
        if args['target_choice_format'] in ["{choice_numeral}. {choice_text}", "{choice_text}"]:
            args['postprocessors'] = [
                "processors.to_string_stripped",
                "processors.take_first_non_empty_line",
                "processors.match_closest_option"
            ]
        template = MultipleChoiceTemplate(**args)
        return template


if __name__ == "__main__":
    # Base arguments for all templates
    dataset_names_to_templates = ConfigParams.dataset_names_to_templates
    override_options = ConfigParams.override_options
    for dataset_name, base_args in dataset_names_to_templates.items():
        # Override options for different parameters and create templates
        generator = MultipleChoiceTemplateGenerator(base_args, override_options)
        created_templates = generator.create_templates()

        # Save templates to local catalog
        catalog_manager = CatalogManager(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH / dataset_name)
        for i, template in tqdm(enumerate(created_templates)):
            catalog_manager.save_to_catalog(template, f"template_{i}")
