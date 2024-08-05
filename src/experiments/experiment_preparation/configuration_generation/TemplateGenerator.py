from abc import abstractmethod
from itertools import product
from typing import List

import pandas as pd
from tqdm import tqdm
from unitxt.templates import Template

from src.experiments.experiment_preparation.datasets_configurations.BaseDatasetConfig import BaseDatasetConfig
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class TemplateGenerator:
    def __init__(self, dataset_config: BaseDatasetConfig, override_options: dict, input_format_func: callable = None):
        """
        Initializes the TemplateGenerator with base arguments and override options.

        @param base_args: A dictionary containing base arguments for the template.
        @param override_options: A dictionary containing override options for the template.
        """
        self.dataset_config = dataset_config
        self.override_options = override_options
        self.input_format_func = input_format_func

    def create_templates(self) -> list:
        """
        Creates a list of MultipleChoiceTemplate instances with different parameters.

        @return: A list of the created templates.
        """
        templates = []
        for options in tqdm(product(*self.override_options.values())):
            override_args = dict(zip(self.override_options.keys(), options))
            override_args['input_format_func'] = self.input_format_func
            template = self.create_template(**override_args)
            templates.append(template)
            print(f"Created template with options: {override_args}")
        print("All templates created!")
        return templates

    @staticmethod
    def create_metadata_from_templates(templates: List[Template], params: dict) -> pd.DataFrame:
        """
        Creates a DataFrame containing the templates and their parameters.

        @param templates: A list of templates.
        @param params: A dictionary containing the parameters of the templates.
        @return: pd.DataFrame: A DataFrame containing the templates and their parameters.
        """
        # create a df that contains the templates and their parameter

        columns = list(params.keys())
        # del columns[columns.index('shuffle_choices')]
        rows = []
        for i, template in tqdm(enumerate(templates)):
            row = []
            row.append(f"template_{i}")
            template_params = [getattr(template, column) for column in columns]
            row.extend(template_params)
            rows.append(row)
        df = pd.DataFrame(rows, columns=['template_name'] + columns)
        df.set_index('template_name', inplace=True)
        return df

    @abstractmethod
    def create_template(self, **override_args) -> Template:
        """
        Creates a MultipleChoiceTemplate instance with specific parameters.

        @param override_args: A dictionary containing override options for the template.

        @return: MultipleChoiceTemplate: The created template instance.
        """
        pass

    @abstractmethod
    def create_and_process_metadata(self, created_templates: List[Template], dataset_name: str,
                                    override_options: dict) -> None:
        """
        Creates and processes metadata for the templates.

        @param created_templates: A list of created templates.
        @param dataset_name: The name of the dataset.
        @param override_options: A dictionary containing override options for the template.
        @return: None
        """
        pass
