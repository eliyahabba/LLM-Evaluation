import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.Utils import Utils

access_token = Utils.get_access_token()

trust_remote_code = True
load_in_4bit = False
load_in_8bit = False
# Define the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with the actual model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=access_token,
                                          padding_side="left",
                                          trust_remote_code=trust_remote_code)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             token=access_token,
trust_remote_code = trust_remote_code,

load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)


# Function to calculate perplexity
def calculate_perplexity(texts, model, tokenizer):
    # Tokenize the input text
    encoded_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded_inputs.input_ids

    # Get model outputs
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Calculate the cross-entropy loss
    loss = outputs.loss

    # Calculate perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

# Example text
text = "Your example text goes here."

# Calculate and print perplexity
# List of example texts
texts = ["Your example text goes here.", "Another example text."]
perplexities = calculate_perplexity(text, model, tokenizer)

for text, perplexity in zip(texts, perplexities):
    print(f"Text: {text}\nPerplexity: {perplexity.item()}\n")
