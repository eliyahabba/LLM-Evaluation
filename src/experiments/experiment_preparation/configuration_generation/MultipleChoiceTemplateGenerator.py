from typing import List

from termcolor import colored
from tqdm import tqdm
from unitxt.templates import MultipleChoiceTemplate

from src.experiments.data_loading.CatalogManager import CatalogManager
from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
from src.experiments.experiment_preparation.configuration_generation.TemplateGenerator import TemplateGenerator
from src.experiments.experiment_preparation.datasets_configurations.DatasetConfigFactory import DatasetConfigFactory
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
        dataset_config = self.dataset_config(override_args)
        if dataset_config.enumerator == 'roman':
            dataset_config.postprocessors = [
                "processors.to_string_stripped",
                "processors.take_first_non_empty_line",
                "processors.match_closest_option"
            ]

        if dataset_config.target_choice_format in ["{choice_numeral}. {choice_text}", "{choice_text}"]:
            dataset_config.postprocessors = [
                "processors.to_string_stripped",
                "processors.take_first_non_empty_line",
                "processors.match_closest_option"
            ]
        template = MultipleChoiceTemplate(**dataset_config.to_dict())
        return template

    def create_and_process_metadata(self, created_templates: List[MultipleChoiceTemplate], dataset_name: str,
                                    override_options: dict) -> None:
        metadata_df = generator.create_metadata_from_templates(created_templates, params=override_options)

        # replace the spaces and new lines with the escape character
        metadata_df['choices_separator'] = metadata_df['choices_separator'].replace(' ', '\\s')
        metadata_df['choices_separator'] = metadata_df['choices_separator'].replace('\n', '\\n')
        # replace the enumerator values with their names
        # convert the enumerator to string to be able to replace the values with their names
        metadata_df['enumerator'] = metadata_df['enumerator'].astype(str)
        metadata_df.replace({"enumerator": ConfigParams.map_enumerator}, inplace=True)
        # save the metadata to a csv file
        metadata_df.to_csv(
            TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH / dataset_name / TemplatesGeneratorConstants.TEMPLATES_METADATA)


if __name__ == "__main__":
    # Base arguments for all templates
    dataset_names_to_configs = DatasetConfigFactory.get_all_datasets()
    override_options = ConfigParams.override_options
    for dataset_name, datasetConfig in dataset_names_to_configs.items():
        print(colored(f"Creating templates for {dataset_name}", "blue"))
        # Override options for different parameters and create templates
        generator = MultipleChoiceTemplateGenerator(datasetConfig, override_options)
        created_templates = generator.create_templates()

        # Save templates to local catalog
        catalog_manager = CatalogManager(TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH / dataset_name)
        for i, template in tqdm(enumerate(created_templates)):
            catalog_manager.save_to_catalog(template, f"template_{i}")

        # add a df that contains the templates and their parameter
        generator.create_and_process_metadata(created_templates, dataset_name, override_options)
