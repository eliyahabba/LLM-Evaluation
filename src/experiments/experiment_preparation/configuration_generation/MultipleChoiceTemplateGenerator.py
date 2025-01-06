import argparse
from dataclasses import asdict
from typing import List

from termcolor import colored
from tqdm import tqdm
from unitxt.templates import MultipleChoiceTemplate

from src.experiments.data_loading.CatalogManager import CatalogManager
from src.experiments.experiment_preparation.configuration_generation.ConfigParams import ConfigParams
from src.experiments.experiment_preparation.configuration_generation.TemplateGenerator import TemplateGenerator
from src.experiments.experiment_preparation.datasets_configurations.InputTemplatesConfigs.MultipleChoiceTemplateConfig import \
    MultipleChoiceTemplateConfigFactory
from src.experiments.experiment_preparation.datasets_configurations.old.DatasetConfigFactory import DatasetConfigFactory
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
        input_config = MultipleChoiceTemplateConfigFactory.create({
            **override_args,
        })
        if input_config.enumerator == 'roman':
            input_config.postprocessors = [
                "processors.to_string_stripped",
                "processors.take_first_non_empty_line",
                "processors.match_closest_option"
            ]

        if input_config.target_choice_format in ["{choice_numeral}. {choice_text}", "{choice_text}"]:
            input_config.postprocessors = [
                "processors.to_string_stripped",
                "processors.take_first_non_empty_line",
                "processors.match_closest_option"
            ]
        template = MultipleChoiceTemplate(**asdict(input_config))
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
        metadata_df.to_csv(TemplatesGeneratorConstants.TEMPLATES_METADATA_PATH)


if __name__ == "__main__":
    # Base arguments for all templates
    catalog_path = TemplatesGeneratorConstants.CATALOG_PATH
    parser = argparse.ArgumentParser(description='Generate multiple choice templates')
    override_options = ConfigParams.override_options
    dataset_names_to_configs = DatasetConfigFactory.get_all_instruct_prompts()
    # for input_format_func, data_folder in zip(input_format_funcs, data_folders):
    for dataset_name, datasetConfig in dataset_names_to_configs.items():
        if "mmlu" not in dataset_name:
            continue
        prompts_instruct_data = datasetConfig.get_all_prompts()
        for prompts_instruct in prompts_instruct_data:
            instruct_folder_name = prompts_instruct.name
            input_format = prompts_instruct.text
            try:
                print(colored(f"Creating templates for {dataset_name}", "blue"))
                # Override options for different parameters and create templates
                generator = MultipleChoiceTemplateGenerator(datasetConfig, override_options, input_format)
                created_templates = generator.create_templates()

                # Save templates to local catalog
                catalog_manager = CatalogManager(catalog_path)
                for template_name, template in tqdm(created_templates.items()):
                    catalog_manager.save_to_catalog(template, f"{dataset_name}.{instruct_folder_name}.{template_name}")

                # add a df that contains the templates and their parameter
                generator.create_and_process_metadata(created_templates.values(), dataset_name, override_options)
                print(colored(f"Templates for {dataset_name} created successfully", "green"))
                break
            except Exception as e:
                print(colored(
                    f"Error in creating templates for {dataset_name} with function {input_format}: {e}",
                    "red"))
                continue
