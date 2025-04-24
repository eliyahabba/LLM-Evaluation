# Multiple Choice Template Generator

A module for generating systematic variations of multiple-choice question templates, used in the DOVE dataset creation pipeline.

## Overview

The `MultipleChoiceTemplateGenerator` creates templates with controlled variations along multiple dimensions:
- Instruction phrasing (different ways to phrase the task)
- Choice separators (newline, comma, etc.)
- Enumerators (letters, numbers, etc.) 
- Choice ordering methods

## Components

### MultipleChoiceTemplateGenerator

Main class for template generation:

```python
def create_template(**override_args) -> MultipleChoiceTemplate:
    """Creates a single template with specified parameters"""

def create_templates() -> Dict[str, Template]:
    """Creates all template variations based on configuration options"""

def create_and_process_metadata(templates, dataset_name, options) -> None:
    """Processes and saves template metadata to CSV"""
```

### Template Variation Dimensions

Defines the template variation dimensions in `TemplateVariationDimensions`:

1. **Instruction Phrasing**:
   - Dataset-specific instruction variations
   - Different ways to phrase the multiple-choice task
   - Controlled through `instruction_phrasing_data`

2. **Enumerators**:
   - `capitals`: A, B, C, ...
   - `lowercase`: a, b, c, ...
   - `numbers`: 1, 2, 3, ...
   - `roman`: I, II, III, ...
   - Special characters (Greek/keyboard)

3. **Choice Separators**:
   - Newline (`\n`)
   - Comma (`, `)
   - Semicolon (`; `)
   - Pipe (` | `)
   - OR variants
   - Space

4. **Choice Ordering**:
   - Original order
   - Length-based sorting
   - Alphabetical sorting
   - Position-based (first/fourth)
   - Reverse ordering

## Usage

```python
from MultipleChoiceTemplateGenerator import MultipleChoiceTemplateGenerator
from TemplateVariationDimensions import TemplateVariationDimensions
from CatalogManager import CatalogManager

# Get dataset configuration with instruction phrasings
dataset_config = DatasetConfigFactory.get_instruct_prompts("MMLU")
instruction_phrasings = dataset_config.get_instruction_phrasings()

# For each instruction phrasing
for instruction in instruction_phrasings:
    # Initialize generator
    generator = MultipleChoiceTemplateGenerator(
        input_config=dataset_config,
        template_dimensions=TemplateVariationDimensions.template_dimensions,
        input_format=instruction.text
    )

    # Generate all template variations
    templates = generator.create_templates()

    # Save to catalog
    catalog_manager = CatalogManager(catalog_path)
    for name, template in templates.items():
        catalog_manager.save_to_catalog(
            template, 
            f"dataset.{instruction.name}.{name}"
        )
```

## Template Naming

Templates are named based on their configuration:
```
{instruction_name}.enumerator_[type]_choicesSeparator_[type]_shuffleChoices_[type]
```

Example: `formal.enumerator_numbers_choicesSeparator_newline_shuffleChoices_lengthSort`

## Metadata

Generates a CSV file with template metadata including:
- Instruction phrasing variations
- Parameter configurations
- Formatting options
- Choice ordering methods

## Integration

This module is part of the DOVE dataset creation pipeline, used to generate systematic prompt variations for evaluating LLM robustness across different formatting dimensions. 