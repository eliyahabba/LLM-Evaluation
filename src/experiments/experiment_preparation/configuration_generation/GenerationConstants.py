class GenerationConstants:
    """Constants used in template generation."""
    
    # Paths
    MULTIPLE_CHOICE_PATH = "MultipleChoiceTemplates"
    DATA_PATH = "Data"
    CATALOG_PATH = "Catalog"
    TEMPLATES_METADATA = "templates_metadata.csv"
    DATASET_SIZES_PATH = "datasets_sizes.csv"
    MMLU_METADATA_PATH = "mmlu_metadata.csv"
    
    # Template types
    MULTIPLE_CHOICE_STRUCTURED = "MultipleChoiceTemplatesStructured"
    MULTIPLE_CHOICE_WITH_TOPIC = "MultipleChoiceTemplatesWithTopic"
    MULTIPLE_CHOICE_INSTRUCTIONS = "MultipleChoiceTemplatesInstructions"
    MULTIPLE_CHOICE_INSTRUCTIONS_WITH_TOPIC = "MultipleChoiceTemplatesInstructionsWithTopic"
    MULTIPLE_CHOICE_INSTRUCTIONS_WITHOUT_TOPIC = "MultipleChoiceTemplatesInstructionsWithoutTopic" 