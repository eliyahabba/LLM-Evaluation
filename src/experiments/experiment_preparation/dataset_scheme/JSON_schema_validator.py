import json
from jsonschema import validate, ValidationError


def validate_json_file(json_file_path, schema_file_path):
    """
    Validate a JSON file against a JSON schema.

    Args:
        json_file_path (str): Path to the JSON file to validate
        schema_file_path (str): Path to the JSON schema file

    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)

        # Read the schema file
        with open(schema_file_path, 'r') as file:
            schema_data = json.load(file)

        # Validate the JSON against the schema
        if isinstance(json_data, list):
            for idx, item in enumerate(json_data):
                validate(instance=item, schema=schema_data)
        else:
            assert isinstance(json_data, dict)
            validate(instance=json_data, schema=schema_data)
        return True, "JSON is valid according to the schema."

    except FileNotFoundError as e:
        return False, f"File not found: {str(e)}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}"
    except ValidationError as e:
        return False, f"Schema validation error: {str(e)}"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    json_file = "data_sample.json"
    json_file = "converted_data.json"
    schema_file = "scheme.json"

    is_valid, message = validate_json_file(json_file, schema_file)
    print(message)