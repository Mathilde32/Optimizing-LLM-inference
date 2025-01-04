# Evaluation Script for Quantized vs. Non-Quantized LLAMA Models

# Import Libraries
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model Names
quantized_model_dir = "./llama-quantized"  # Quantized model path
non_quantized_model_name = "meta-llama/Llama-2-7b-hf"  # Non-quantized model

# Load Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(non_quantized_model_name)

# Load Non-Quantized Model
print("Loading non-quantized model...")
non_quantized_model = AutoModelForCausalLM.from_pretrained(non_quantized_model_name, device_map="auto")

# Load Quantized Model
print("Loading quantized model...")
quantized_model = AutoModelForCausalLM.from_pretrained(quantized_model_dir, device_map="auto")

# Evaluation Function
def evaluate_model(model, prompt, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    start_time = time.time()
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    elapsed_time = time.time() - start_time
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, elapsed_time

# Test Prompt
prompt = "The impact of technology on society is "

# Evaluate Non-Quantized Model
print("Evaluating non-quantized model...")
non_quantized_text, non_quantized_time = evaluate_model(non_quantized_model, prompt, tokenizer)
print(f"Non-Quantized Model Output:\n{non_quantized_text}")
print(f"Time Taken: {non_quantized_time:.2f} seconds\n")

# Evaluate Quantized Model
print("Evaluating quantized model...")
quantized_text, quantized_time = evaluate_model(quantized_model, prompt, tokenizer)
print(f"Quantized Model Output:\n{quantized_text}")
print(f"Time Taken: {quantized_time:.2f} seconds\n")

# Compare Results
print("Comparison of Models:")
print(f"Non-Quantized Model Time: {non_quantized_time:.2f} seconds")
print(f"Quantized Model Time: {quantized_time:.2f} seconds")
