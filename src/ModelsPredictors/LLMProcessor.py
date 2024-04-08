import argparse
from typing import Union, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

from src.utils.Constants import Constants
from src.utils.ReadLLMParams import ReadLLMParams
from src.utils.Utils import Utils

LLMProcessorConstants = Constants.LLMProcessorConstants
access_token = Utils.get_access_token()


class LLMProcessor:
    def __init__(self, model_name: str,
                 load_in_4bit: bool = False, load_in_8bit: bool = False,
                 trust_remote_code: bool = False, return_token_type_ids: bool = True):
        # Define the pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       token=access_token,
                                                       padding_side="left",
                                                       trust_remote_code=trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Most LLMs don't have a pad token by default
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          load_in_4bit=load_in_4bit,
                                                          load_in_8bit=load_in_8bit,
                                                          device_map="auto",
                                                          token=access_token,
                                                          trust_remote_code=trust_remote_code)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_token_type_ids = return_token_type_ids

    def tokenize_text(self, input_text: str) -> BatchEncoding:
        """
        Tokenize the input text.

        @param input_text: Text to be tokenized.
        @return: Tokenized input text.
        """
        return self.tokenizer(input_text, return_tensors="pt", return_token_type_ids=self.return_token_type_ids,
                              padding=True).to(self.device)

    def generate_text(self, input_tokenized: BatchEncoding, max_new_tokens: int = 5) -> dict:
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

    def print_generated_tokens(self, generated_tokens: torch.Tensor, transition_scores: torch.Tensor) -> None:
        """
        Print generated tokens with their scores and probabilities.

        @param generated_tokens: Generated token sequences.
        @param transition_scores: Transition scores associated with generated tokens.
        """
        generated_tokens = generated_tokens.cpu()
        transition_scores = transition_scores.cpu()
        print("The generated tokens with their scores and probabilities are:")
        print("| token | token string | logits | probability")
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            print(
                f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}"
            )

    def print_generated_tokens_decoded(self, generated_tokens_decoded: List[str]) -> None:
        """
        Print decoded generated tokens.

        @param generated_tokens_decoded: Decoded generated tokens.
        """
        print("The decoded generated tokens are:")
        print(generated_tokens_decoded)

    def generate_model_text(self, input_text: Union[str, List[str]], max_new_tokens: int,
                            is_print: bool = False) -> str:
        """
        Generate text using a pre-trained language model and print the results.

        @param input_text: Text used as input for text generation.
        @param is_print: Whether to print the generated tokens and their scores and probabilities.

        @return: The generated tokens decoded.
        """
        input_tokenized = self.tokenize_text(input_text)
        outputs = self.generate_text(input_tokenized, max_new_tokens)
        generated_tokens = outputs.sequences[:, input_tokenized.input_ids.shape[1]:]
        generated_tokens_decoded = self.decode_tokens(generated_tokens)
        if is_print:
            transition_scores = self.compute_transition_scores(outputs.sequences, outputs.scores)
            self.print_generated_tokens(generated_tokens, transition_scores)
            self.print_generated_tokens_decoded(generated_tokens_decoded)
        return generated_tokens_decoded

    def predict(self, input_text: Union[str, List[str]], max_new_tokens: int, is_print: bool = False) -> str:
        """
        Predict the next word in the sequence.
        """
        return self.generate_model_text(input_text, max_new_tokens, is_print=is_print)


# Execute the main function
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = ReadLLMParams.read_llm_params(args)
    args.add_argument("--batch_size", type=int, default=2)
    args = args.parse_args()
    model_name = args.model_name
    llmp = LLMProcessor(model_name=model_name, load_in_4bit=args.not_load_in_4bit, load_in_8bit=args.not_load_in_8bit,
                        trust_remote_code=args.trust_remote_code, return_token_type_ids=args.not_return_token_type_ids)
    sentences = ["please tell about the history of the world.",
                 "please tell about the world cup history."]
    llmp.predict(sentences, max_new_tokens=5)
