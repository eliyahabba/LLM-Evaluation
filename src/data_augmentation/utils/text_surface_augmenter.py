# Non-semantic changes / structural changes (UNI TEXT)
import itertools
import random
import re
from typing import List

import numpy as np
from src.data_augmentation.config.constants import GLOBAL_RANDOM_SEED


# Constants for NonLLMAugmenter
class TextSurfaceAugmenterConstants:
    # White space options
    WHITE_SPACE_OPTIONS = ["\n", "\t", " ", ""]

    # Keyboard layout for butter finger
    QUERTY_KEYBOARD = {
        "q": "qwasedzx",
        "w": "wqesadrfcx",
        "e": "ewrsfdqazxcvgt",
        "r": "retdgfwsxcvbnju",
        "t": "tryfhgedcvbnju",
        "y": "ytugjhrfvbnji",
        "u": "uyihkjtgbnmlo",
        "i": "iuojlkyhnmlp",
        "o": "oipklujm",
        "p": "plo['ik",
        "a": "aqszwxwdce",
        "s": "swxadrfv",
        "d": "decsfaqgbv",
        "f": "fdgrvwsxyhn",
        "g": "gtbfhedcyjn",
        "h": "hyngjfrvkim",
        "j": "jhknugtblom",
        "k": "kjlinyhn",
        "l": "lokmpujn",
        "z": "zaxsvde",
        "x": "xzcsdbvfrewq",
        "c": "cxvdfzswergb",
        "v": "vcfbgxdertyn",
        "b": "bvnghcftyun",
        "n": "nbmhjvgtuik",
        "m": "mnkjloik",
        " ": " "
    }

    PUNCTUATION_MARKS = [".", ",", "!", "?", ";", ":", "-", "_"]

    # Default probabilities
    DEFAULT_TYPO_PROB = 0.05
    DEFAULT_CASE_CHANGE_PROB = 0.1

    # Default max outputs
    DEFAULT_MAX_OUTPUTS = 1

    # Random ranges for white space generation
    MIN_WHITESPACE_COUNT = 1
    MAX_WHITESPACE_COUNT = 3

    # Random index range for white space options
    MIN_WHITESPACE_INDEX = 0
    MAX_WHITESPACE_INDEX = 2

    # Transformation techniques
    TRANSFORMATION_TECHNIQUES = ["typos", "capitalization", "punctuation", "spacing"]

class TextSurfaceAugmenter:
    """
    Augmenter that creates variations of prompts using non-LLM techniques.
    This includes simple transformations like adding typos, changing capitalization, etc.
    """

    def __init__(self, n_augments=3, random_seed=GLOBAL_RANDOM_SEED):
        """
        Initialize the non-LLM augmenter.

        Args:
            n_augments: Number of variations to generate
            random_seed: Random seed for deterministic behavior
        """
        self.n_augments = n_augments
        self.random_seed = random_seed
        # Initialize random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def _add_white_spaces_to_single_text(self, value, probability=0.8, min_spaces=1, max_spaces=3, word_probability=1.0):
        """
        Add white spaces to the input text.

        Args:
            value: The input text to augment.
            probability: Probability of replacing a space with random whitespace (0-1)
            min_spaces: Minimum number of whitespace characters to add when replacing a space
            max_spaces: Maximum number of whitespace characters to add when replacing a space
            word_probability: Probability of selecting a word for whitespace modification (0-1)

        Returns:
            Augmented text with added white spaces.
        """
        # Reseed for deterministic behavior
        random.seed(self.random_seed)
        
        words = re.split(r"(\s+)", value)
        new_value = ""

        for word in words:
            if word.isspace() and random.random() < probability and random.random() < word_probability:
                # Generate a random number of whitespace characters based on min_spaces and max_spaces
                for j in range(random.randint(min_spaces, max_spaces)):
                    new_value += TextSurfaceAugmenterConstants.WHITE_SPACE_OPTIONS[random.randint(
                        TextSurfaceAugmenterConstants.MIN_WHITESPACE_INDEX,
                        TextSurfaceAugmenterConstants.MAX_WHITESPACE_INDEX)]
            else:
                new_value += word
        return new_value

    def add_white_spaces(self, inputs, probability=0.8, min_spaces=1, max_spaces=3, word_probability=1.0, preserve_original=False):
        """
        Add white spaces to input text.

        Args:
            inputs: A text string to augment.
            probability: Probability of replacing each space with random whitespace (0-1).
                Set to 0 to keep original text without changes.
            min_spaces: Minimum number of whitespace characters to add when replacing a space
            max_spaces: Maximum number of whitespace characters to add when replacing a space
            word_probability: Probability of selecting a word for whitespace modification (0-1)
            preserve_original: If True, returns the original text without changes.

        Returns:
            A string with augmented spaces if not preserve_original, otherwise the original text.
        """
        # Reseed for deterministic behavior
        random.seed(self.random_seed)
        
        # Return original text if preserve_original is True or probability is 0
        if preserve_original or probability <= 0:
            return inputs

        # Process single text input and return a single result
        return self._add_white_spaces_to_single_text(inputs, probability, min_spaces, max_spaces, word_probability)

    def butter_finger(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, keyboard="querty", seed=0,
                      max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Introduce typos in the text by simulating butter fingers on a keyboard.

        Args:
            text: Input text to augment.
            prob: Probability of introducing a typo for each character.
            keyboard: Keyboard layout to use.
            seed: Random seed for reproducibility.
            max_outputs: Maximum number of augmented outputs.

        Returns:
            List of texts with typos.
        """
        # Use the instance seed combined with the method seed for consistent but varying outputs
        combined_seed = self.random_seed + seed
        random.seed(combined_seed)
        
        key_approx = TextSurfaceAugmenterConstants.QUERTY_KEYBOARD if keyboard == "querty" else {}

        if not key_approx:
            print("Keyboard not supported.")
            return [text]

        prob_of_typo = int(prob * 100)
        perturbed_texts = []
        for _ in itertools.repeat(None, max_outputs):
            butter_text = ""
            for letter in text:
                lcletter = letter.lower()
                if lcletter not in key_approx.keys():
                    new_letter = lcletter
                else:
                    if random.choice(range(0, 100)) <= prob_of_typo:
                        new_letter = random.choice(key_approx[lcletter])
                    else:
                        new_letter = lcletter
                # go back to original case
                if not lcletter == letter:
                    new_letter = new_letter.upper()
                butter_text += new_letter
            perturbed_texts.append(butter_text)
        return perturbed_texts

    def change_char_case(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_CASE_CHANGE_PROB, seed=0,
                         max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Change the case of characters in the text.

        Args:
            text: Input text to augment.
            prob: Probability of changing the case of each character.
            seed: Random seed for reproducibility.
            max_outputs: Maximum number of augmented outputs.

        Returns:
            List of texts with modified character cases.
        """
        # Use the instance seed combined with the method seed for consistent but varying outputs
        combined_seed = self.random_seed + seed
        random.seed(combined_seed)
        
        results = []
        for _ in range(max_outputs):
            result = []
            for c in text:
                if c.isupper() and random.random() < prob:
                    result.append(c.lower())
                elif c.islower() and random.random() < prob:
                    result.append(c.upper())
                else:
                    result.append(c)
            result = "".join(result)
            results.append(result)
        return results


    def swap_characters(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, seed=0,
                        max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Swaps characters in text, with probability prob for ang given pair.
        Ex: 'apple' -> 'aplpe'
        Arguments:
            text (string): text to transform
            prob (float): probability of any two characters swapping. Default: 0.05
            seed (int): random seed
            max_outputs: Maximum number of augmented outputs.
            (taken from the NL-Augmenter project)
        """
        # Use the instance seed combined with the method seed for consistent but varying outputs
        combined_seed = self.random_seed + seed
        
        results = []
        for _ in range(max_outputs):
            max_seed = 2 ** 32
            # seed with hash so each text of same length gets different treatment.
            np.random.seed((combined_seed + sum([ord(c) for c in text])) % max_seed)
            # number of possible characters to swap.
            num_pairs = len(text) - 1
            # if no pairs, do nothing
            if num_pairs < 1:
                return text
            # get indices to swap.
            indices_to_swap = np.argwhere(
                np.random.rand(num_pairs) < prob
            ).reshape(-1)
            # shuffle swapping order, may matter if there are adjacent swaps.
            np.random.shuffle(indices_to_swap)
            # convert to list.
            text = list(text)
            # swap.
            for index in indices_to_swap:
                text[index], text[index + 1] = text[index + 1], text[index]
            # convert to string.
            text = "".join(text)
            results.append(text)
        return results

    def switch_punctuation(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, seed=0, max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Switches punctuation in text with a probability of prob.
        Arguments:
            text (string): text to transform
            prob (float): probability of any two characters switching. Default: 0.05
            seed (int): random seed
            max_outputs: Maximum number of augmented outputs.
        """
        # Use the instance seed combined with the method seed for consistent but varying outputs
        combined_seed = self.random_seed + seed
        np.random.seed(combined_seed)
        
        results = []
        for _ in range(max_outputs):
            text_chars = list(text)
            for i in range(len(text_chars)):
                if text_chars[i] in TextSurfaceAugmenterConstants.PUNCTUATION_MARKS and np.random.rand() < prob:
                    # Randomly select a different punctuation mark to switch with
                    new_punctuation = np.random.choice([p for p in TextSurfaceAugmenterConstants.PUNCTUATION_MARKS
                                                        if p != text_chars[i]])
                    text_chars[i] = new_punctuation
            results.append("".join(text_chars))
        return results

    def augment(self, text: str, techniques: List[str] = None) -> List[str]:
        """
        Apply text surface transformations to generate variations.

        Args:
            text: The text to augment
            techniques: List of techniques to apply in sequence. If None, a default sequence will be used.
                Options: "typos", "capitalization", "spacing", "no_spacing", "swap_characters", "punctuation", "no_transform"

        Returns:
            List of augmented texts including the original text
        """
        # Reseed for deterministic behavior
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Default sequence if none provided
        if techniques is None:
            techniques = ["typos", "capitalization", "spacing", "no_spacing", "swap_characters", "punctuation"]
            
        # If "no_transform" is in techniques, return only the original text
        if "no_transform" in techniques:
            return [text]

        # Start with the original text
        variations = [text]

        # Apply each technique in sequence
        for technique in techniques:
            new_variations = []

            # Always keep the original variations
            new_variations.extend(variations)

            # For each existing variation, apply the current technique
            for variation in variations:
                if technique == "typos":
                    # Add typo variations
                    typo_results = self.butter_finger(variation, prob=0.1, max_outputs=2)
                    new_variations.extend(typo_results)
                elif technique == "capitalization":
                    # Add case variations
                    case_results = self.change_char_case(variation, prob=0.15, max_outputs=2)
                    new_variations.extend(case_results)
                elif technique == "spacing":
                    # Add spacing variations with more sophisticated parameters
                    spacing_result1 = self.add_white_spaces(
                        variation, 
                        probability=0.8,
                        min_spaces=1, 
                        max_spaces=3, 
                        word_probability=0.7
                    )
                    
                    spacing_result2 = self.add_white_spaces(
                        variation, 
                        probability=0.6,
                        min_spaces=2, 
                        max_spaces=4, 
                        word_probability=0.5
                    )
                    
                    new_variations.append(spacing_result1)
                    new_variations.append(spacing_result2)
                
                elif technique == "no_spacing":
                    # Explicitly preserve original spacing
                    no_spacing_result = self.add_white_spaces(variation, preserve_original=True)
                    new_variations.append(no_spacing_result)
                    
                elif technique == "swap_characters":
                    # Add character swap variations
                    swap_results = self.swap_characters(variation, max_outputs=2)
                    new_variations.extend(swap_results)
                elif technique == "punctuation":
                    # Add punctuation variations
                    punctuation_results = self.switch_punctuation(variation, max_outputs=2)
                    new_variations.extend(punctuation_results)

            # Update variations for the next technique
            variations = new_variations

            # If we already have enough variations, we can stop
            if len(variations) >= self.n_augments:
                break

        # Remove duplicates while preserving order
        unique_variations = []
        for var in variations:
            if var not in unique_variations:
                unique_variations.append(var)

        # Ensure we return the requested number of variations
        if len(unique_variations) > self.n_augments:
            # Keep the original text and sample from the rest
            original = unique_variations[0]
            rest = unique_variations[1:]
            # Use our global seed for consistent sampling
            random.seed(self.random_seed)
            sampled = random.sample(rest, min(self.n_augments - 1, len(rest)))
            return [original] + sampled

        return unique_variations


if __name__ == "__main__":
    # Create the augmenter
    augmenter = TextSurfaceAugmenter(n_augments=5)

    # Example 1: Simple text with default sequence
    text1 = "This is a simple example of text surface augmentation."
    text1_1 = "This, is a simple example: Text surface augmentation."
    variations1 = augmenter.augment(text1)

    print(f"Original text: {text1}")
    print(f"\nGenerated {len(variations1)} variations with default sequence:")
    for i, variation in enumerate(variations1):
        if variation == text1:
            print(f"\nVariation {i+1} (Original):")
        else:
            print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50)

    # Example 2: Custom sequence
    text2 = "What is the capital of France? Paris is the correct answer."
    variations2 = augmenter.augment(text2, techniques=["spacing", "typos"])

    print(f"\nOriginal text: {text2}")
    print(f"\nGenerated {len(variations2)} variations with custom sequence (spacing → typos):")
    for i, variation in enumerate(variations2):
        if variation == text2:
            print(f"\nVariation {i+1} (Original):")
        else:
            print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50)

    # Example 3: Individual transformations
    print("\nIndividual transformations:")
    print(f"Original: {text1}")
    print(f"With typos: {augmenter.butter_finger(text1, prob=0.1, max_outputs=1)[0]}")
    print(f"With capitalization changes: {augmenter.change_char_case(text1, prob=0.15, max_outputs=1)[0]}")
    
    # Updated spacing example
    print(f"With spacing changes (prob=0.8): {augmenter.add_white_spaces(text1, probability=0.8)}")
    print(f"With spacing changes (prob=0.3): {augmenter.add_white_spaces(text1, probability=0.3)}")
    print(f"Without spacing changes (preserve original): {augmenter.add_white_spaces(text1, preserve_original=True)}")
    
    print(f"With character swaps: {augmenter.swap_characters(text1, prob=0.08, max_outputs=1)[0]}")
    print(f"With punctuation changes: {augmenter.switch_punctuation(text1_1, prob=0.9, max_outputs=1)[0]}")
