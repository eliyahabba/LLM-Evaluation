import random

random.seed(42)


class ConfigParams:
    override_options = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman"],
        "choices_separator": [" ", "\n", ", ", "; ", " | ", " OR ", " or "],
        "shuffle_choices": [False, True],
        # Add more parameters and their possible values as needed
    }

    map_enumerator = {"ABCDEFGHIJKLMNOP": "capitals",
                      "abcdefghijklmnop": "lowercase",
                      str(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                           '18', '19', '20']): "numbers",
                      str(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
                           'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']): "roman"}
