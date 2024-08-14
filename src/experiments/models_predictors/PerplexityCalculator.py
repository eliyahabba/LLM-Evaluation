import datasets
import numpy as np
import torch
from evaluate import logging
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.Utils import Utils

access_token = Utils.get_access_token()

class PerplexityCalculator:
    def compute(
            self, model, tokenizer, encodings, batch_size: int = 16, add_start_token: bool = True, device=None,
            max_length=None
    ):
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                    len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                    tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                # print(tokenizer.bos_token_id)
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return np.round(ppls, 2)


if __name__ == "__main__":
    # Example usage
    # List of example texts
    trust_remote_code = True
    load_in_4bit = True
    load_in_8bit = False
    # model_name = "allenai/OLMo-1.7-7B-hf"  # Replace with the actual model name if different
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with the actual model name if different

    texts = ["Your example text goes here.", "Another example text."]
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=access_token,
                                              padding_side="left",
                                              trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 load_in_4bit=load_in_4bit,
                                                 load_in_8bit=load_in_8bit,
                                                 device_map="auto",
                                                 token=access_token,
                                                 trust_remote_code=trust_remote_code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encodings = tokenizer(texts, return_tensors="pt", return_token_type_ids=True,
                          padding=True).to(device)

    input_tokenized = datasets.Dataset.from_dict({"input_text": texts})
    perplexity_calculator = PerplexityCalculator()
    perplexity = perplexity_calculator.compute(model, tokenizer, encodings, device=device)
    print(perplexity)
    text = "Your example text goes here."
    perplexity = perplexity_calculator.compute(model, tokenizer, encodings, device=device)
    # Calculate and print perplexity
