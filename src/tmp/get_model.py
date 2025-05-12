from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch
import os
import gc
import time

def log_gpu_memory():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} - Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB, "
                 f"Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB, "
                 f"Max allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")
    else:
        print("No GPU available")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Login to Hugging Face
print("\n[0] Logging in to Hugging Face...")
login(token="YOUR_HF_TOKEN")  # Replace with your actual token
print("Successfully logged in to Hugging Face")

# Configure model loading with 4-bit quantization
model_name = "meta-llama/Llama-3.3-70B-Instruct"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # Load model in 4-bit precision
    bnb_4bit_use_double_quant=True,   # Use double quantization for more memory savings
    bnb_4bit_quant_type="nf4",        # Normal Float 4-bit quantization (better for LLMs)
    bnb_4bit_compute_dtype=torch.float16,  # Compute in half precision
    llm_int8_skip_modules=["lm_head"],    # Skip quantizing specific modules
    llm_int8_threshold=6.0,           # For mixed-precision schemes
    llm_int8_has_fp16_weight=False,   # Save memory by not storing fp16 copy
)

print("\n[1] Loading tokenizer...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")

print("\n[2] Loading model with 4-bit quantization...")
print("Initial GPU memory:")
log_gpu_memory()

start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",              # Automatically distribute across available devices
    low_cpu_mem_usage=True,         # Reduce CPU memory usage during loading
    torch_dtype=torch.float16,      # Use half precision
)
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

print("\n[3] GPU memory after loading:")
log_gpu_memory()

# Test the model
print("\n[4] Testing model with a simple prompt...")
prompt = "Tell me the advantages of quantization in large language models."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

start_time = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
print(f"Generation completed in {time.time() - start_time:.2f} seconds")

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nModel response:\n{response}")

print("\n[5] Final GPU memory:")
log_gpu_memory()

# Save memory usage statistics to a file
with open("memory_usage.txt", "w") as f:
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write(f"CUDA version: {torch.version.cuda}\n")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            f.write(f"GPU {i} - Max allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB\n")