import argparse
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
        metadata_df.replace({"enumerator": ConfigParams.ENUM_CHARS}, inplace=True)
        # save the metadata to a csv file
        metadata_df.to_csv(
            TemplatesGeneratorConstants.MULTIPLE_CHOICE_PATH / dataset_name / TemplatesGeneratorConstants.TEMPLATES_METADATA)


if __name__ == "__main__":
    # Base arguments for all templates

    # add input format to the parser
    data_path = TemplatesGeneratorConstants.DATA_PATH
    catalog_path = TemplatesGeneratorConstants.CATALOG_PATH
    parser = argparse.ArgumentParser(description='Generate multiple choice templates')
    parser.add_argument('--input_format_func', type=str, default="get_mmlu_instructions_with_topic",
                        help='The input format for the templates')
    parser.add_argument('--data_folder', type=str,
                        default="MultipleChoiceTemplatesInstructionsWithTopic",
                        help='The data folder for the templates')
    input_format_funcs = [
        "get_mmlu_instructions_with_topic",
        "get_mmlu_instructions_without_topic",
        "get_mmlu_instructions_with_topic_helm",
        "get_mmlu_instructions_without_topic_helm",
        "get_structured_instruction_with_topic",
        "get_structured_instruction_without_topic"
    ]
        
    data_folders  = \
        ["MultipleChoiceTemplatesInstructionsWithTopic",
        "MultipleChoiceTemplatesInstructionsWithoutTopic",
        "MultipleChoiceTemplatesInstructionsWithTopicHelm",
        "MultipleChoiceTemplatesInstructionsWithoutTopicHelm",
         "MultipleChoiceTemplatesStructuredWithTopic",
         "MultipleChoiceTemplatesStructuredWithoutTopic"]

    args = parser.parse_args()
    dataset_names_to_configs = DatasetConfigFactory.get_all_datasets()
    override_options = ConfigParams.override_options
    for input_format_func, data_folder in zip(input_format_funcs, data_folders):
        for dataset_name, datasetConfig in dataset_names_to_configs.items():
            if "mmlu" not in dataset_name:
                continue
            try:
                print(colored(f"Creating templates for {dataset_name}", "blue"))
                # Override options for different parameters and create templates
                generator = MultipleChoiceTemplateGenerator(datasetConfig, override_options, input_format_func)
                created_templates = generator.create_templates()

                # Save templates to local catalog
                catalog_manager = CatalogManager(catalog_path)
                for template_name, template in tqdm(created_templates.items()):
                    catalog_manager.save_to_catalog(template, f"{data_folder}.{template_name}")

                # add a df that contains the templates and their parameter
                generator.create_and_process_metadata(created_templates.values(), dataset_name, override_options)
                print(colored(f"Templates for {dataset_name} created successfully", "green"))
                break
            except Exception as e:
                print(colored(f"Error in creating templates for {dataset_name} with function {args.input_format_func}: {e}",
                              "red"))
                continue
