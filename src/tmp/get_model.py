from transformers import pipeline

from huggingface_hub import login

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch

# Login to Hugging Face (replace with your token)
login(token="YOUR_TOKEN_HERE")

# Define model name
model_name = "meta-llama/Llama-3.3-70B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision
    load_in_4bit=True,          # Enable 4-bit quantization
    device_map="auto",          # Automatically distribute across available devices
    low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
)

# Example usage
messages = [
    {"role": "user", "content": "מי אתה?"},
]

# Generate response
response = pipe(messages)
print(response[0]['generated_text'])

# For conversation format, you can also use:
# from transformers import TextIteratorStreamer
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "מי אתה?"},
]
response = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(response, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))