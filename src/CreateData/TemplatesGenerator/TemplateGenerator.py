from abc import abstractmethod
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
            template = self.create_template(**override_args)
            templates.append(template)
            print(f"Created template with options: {override_args}")
        print("All templates created!")
        return templates

    @abstractmethod
    def create_template(self, **override_args) -> MultipleChoiceTemplate:
        """
        Creates a MultipleChoiceTemplate instance with specific parameters.

        @param override_args: A dictionary containing override options for the template.

        @return: MultipleChoiceTemplate: The created template instance.
        """
        pass
