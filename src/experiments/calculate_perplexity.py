import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# Define the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with the actual model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
load_in_4bit = False
load_in_8bit = False

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)


# Function to calculate perplexity
def calculate_perplexity(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # Calculate the cross-entropy loss
    loss = outputs.loss

    # Calculate perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

# Example text
text = "Your example text goes here."

# Calculate and print perplexity
perplexity = calculate_perplexity(text, model, tokenizer)
print(f"Perplexity: {perplexity}")
