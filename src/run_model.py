# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # מגדירים את ההתקן כ-MPS
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")
#
# # שם המודל
# model_name = "allenai/OLMoE-1B-7B-0924-Instruct"
#
# # טוען את הטוקניזר
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # טוען את המודל ל-MPS
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16  # ניתן לשנות ל-torch.float32 אם יש בעיות
# ).to(device)
#
# # השאלה למודל
# question = "Could you provide a response to the following question:\nKuathiriwa kwa neva ya uso katika forameni ya stylomastoidi kutasababisha kwa upande huo huo\n1. kupooza kwa misuli ya uso.\n2. kupooza kwa misuli ya uso na kupoteza ladha.\n3. kupooza kwa misuli ya uso, kupoteza ladha, ו kutokwa na machozi.\n4. kupooza kwa misuli ya uso, kupoteza ladha, kutokwa na machozi, ו kupungua kwa utoaji של mate.\nAnswer:"
#
# # טוקניזציה
# inputs = tokenizer(question, return_tensors="pt").to(device)
#
# # הרצת המודל
# with torch.no_grad():
#     output = model.generate(**inputs, max_new_tokens=100)
#
# # פענוח התשובה
# answer = tokenizer.decode(output[0], skip_special_tokens=True)
# print("\nModel's Response:\n", answer)



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ודא שה-GPU של NVIDIA זמין
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# רשימת המודלים להערכה
models_to_evaluate = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'allenai/OLMoE-1B-7B-0924-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]

new_models_to_evaluate = [
    'meta-llama/Llama-3.2-70B-Instruct',
    'nvidia/Mistral-NeMo-12B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct'
]

# שילוב כל המודלים
all_models = models_to_evaluate + new_models_to_evaluate

# השאלה למודל
question = """Could you provide a response to the following question:
Kuathiriwa kwa neva ya uso katika forameni ya stylomastoidi kutasababisha kwa upande huo huo
1. kupooza kwa misuli ya uso.
2. kupooza kwa misuli ya uso na kupoteza ladha.
3. kupooza kwa misuli ya uso, kupoteza ladha, na kutokwa na machozi.
4. kupooza kwa misuli ya uso, kupoteza ladha, kutokwa na machozi, ו kupungua kwa utoaji wa mate.
Answer:"""

# פונקציה להרצת מודל עם כיוונון כמותי 8-bit
def run_model(model_name, question):
    print(f"\nLoading model: {model_name}")

    # קונפיגורציה של כיוונון כמותי ל-8bit
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # טוען טוקניזר
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # טוען מודל עם כיוונון כמותי
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # טוקניזציה
    inputs = tokenizer(question, return_tensors="pt").to(device)

    # הרצת המודל
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    # פענוח והצגת תשובה
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nModel {model_name} response:\n{answer}")

# הרצת כל המודלים
for model in all_models:
    try:
        run_model(model, question)
    except Exception as e:
        print(f"Error with model {model}: {e}")