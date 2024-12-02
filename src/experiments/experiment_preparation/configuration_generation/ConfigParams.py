import itertools
import random
from typing import List, Dict, Any

random.seed(42)


class ConfigParams:
    GREEK_CHARS = "αβγδεζηθικ"  # 10 Greek letters
    KEYBOARD_CHARS = "!@#$%^₪*)("  # 26 lowercase letters
    override_options = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman", KEYBOARD_CHARS, GREEK_CHARS],

        "choices_separator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
        "shuffle_choices": [False, True],
        # Add more parameters and their possible values as needed
    }

    ENUM_CHARS = {"ABCDEFGHIJKLMNOP": "capitals",
                  "abcdefghijklmnop": "lowercase",
                  str(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19', '20']): "numbers",
                  str(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
                       'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']): "roman",
                  KEYBOARD_CHARS: "keyboard",  # Added mapping for keyboard chars
                  GREEK_CHARS: "greek"  # Added mapping for greek chars
                  }

    @classmethod
    def get_options(cls) -> Dict[str, List]:
        return cls.override_options

    @classmethod  # Changed to classmethod to access class attributes
    def format_value(cls, value: Any) -> str:
        """Format a value for use in template name"""
        if isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            # Special characters mapping
            separator_names = {
                " ": "space",
                "\n": "newline",
                " | ": "pipe",
                " OR ": "OrCapital",
                " or ": "orLower",
                ", ": "comma",
                "; ": "semicolon"
            }

            # Check separators first
            if value in separator_names:
                return separator_names[value]

            # Check special character sets
            if value in cls.ENUM_CHARS:
                return cls.ENUM_CHARS[value]

            # For any other string values
            return value.replace(" ", "")

        return str(value)

    @classmethod
    def generate_template_name(cls, combination: Dict[str, Any]) -> str:
        """Generate a descriptive template name from a combination of options"""
        parts = []
        for key, value in combination.items():
            formatted_value = cls.format_value(value)
            camel_key = cls.to_camel_case(key)
            parts.append(f"{camel_key}_{formatted_value}")
        return "_".join(parts)

    @staticmethod
    def to_camel_case(snake_str: str) -> str:
        """Convert snake_case to camelCase"""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    @classmethod
    def generate_template_combinations(cls) -> Dict[str, Dict]:
        """Generate all possible combinations of template options"""
        options = cls.get_options()
        combinations = [
            {key: value for key, value in zip(options.keys(), values)}
            for values in itertools.product(*options.values())
        ]

        return {
            cls.generate_template_name(combo): combo
            for combo in combinations
        }
