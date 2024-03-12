import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding


class LLMProcessor:
    def __init__(self, model_name: str):
        # Define the pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize_text(self, input_text: str) -> BatchEncoding:
        """
        Tokenize the input text.

        @param input_text: Text to be tokenized.
        @return: Tokenized input text.
        """
        return self.tokenizer(input_text, return_tensors="pt")

    def generate_text(self, input_tokenized: BatchEncoding, max_new_tokens: int = 20) -> dict:
        """
        Generate text using a pre-trained language model.

        @param input_tokenized: Tokenized input text.
        @param max_new_tokens: Maximum number of tokens to generate.
        @return: Generated text.
        """
        outputs = self.model.generate(
            **input_tokenized,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        return outputs

    def compute_transition_scores(self, sequences: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute transition scores for generated tokens.

        @param sequences: Generated token sequences.
        @param scores: Scores associated with generated tokens.
        @return: Transition scores.
        """
        return self.model.compute_transition_scores(sequences, scores, normalize_logits=True)

    def decode_tokens(self, generated_tokens):
        """
        Decode generated tokens.

        @param generated_tokens: Generated token sequences.
        @return: Decoded tokens.
        """
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def print_generated_tokens(self, generated_tokens, transition_scores):
        """
        Print generated tokens with their scores and probabilities.

        @param generated_tokens: Generated token sequences.
        @param transition_scores: Transition scores associated with generated tokens.
        """
        print("The generated tokens with their scores and probabilities are:")
        print("| token | token string | logits | probability")
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            print(
                f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}"
            )

    def print_generated_tokens_decoded(self, generated_tokens_decoded):
        """
        Print decoded generated tokens.

        @param generated_tokens_decoded: Decoded generated tokens.
        """
        print("The decoded generated tokens are:")
        print(generated_tokens_decoded)

    def generate_model_text(self, input_text: str, is_print: bool = False) -> list:
        """
        Generate text using a pre-trained language model and print the results.

        @param input_text: Text used as input for text generation.
        @param is_print: Whether to print the generated tokens and their scores and probabilities.

        @return: The generated tokens decoded.
        """
        input_tokenized = self.tokenize_text(input_text)
        outputs = self.generate_text(input_tokenized)
        transition_scores = self.compute_transition_scores(outputs.sequences, outputs.scores)
        generated_tokens = outputs.sequences[:, input_tokenized.input_ids.shape[1]:]
        if is_print:
            self.print_generated_tokens(generated_tokens, transition_scores)
        generated_tokens_decoded = self.decode_tokens(generated_tokens)
        if is_print:
            self.print_generated_tokens_decoded(generated_tokens_decoded)
        return generated_tokens_decoded

    def predict(self, input_text: str):
        """
        Predict the next word in the sequence.
        """
        return self.generate_model_text(input_text)


# Execute the main function
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    args = args.parse_args()
    model_name = args.model_name
    llmp = LLMProcessor(model_name)
    llmp.predict("please tell about the history of the world.")
