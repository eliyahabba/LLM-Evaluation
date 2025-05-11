from transformers import pipeline

from huggingface_hub import login
login(token="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")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")
pipe(messages)