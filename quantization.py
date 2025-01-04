# Quantization Script for LLAMA Models

# Import Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb

# Model Name and Tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Adjust as needed
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the Model in 8-bit Precision
print(f"Loading {model_name} in 8-bit precision...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Enable 8-bit quantization
    device_map="auto"   # Use available devices (e.g., GPU)
)

# Verify Quantized Model
print("Model successfully loaded and quantized:")
print(model)

# Save Quantized Model
output_dir = "./llama-quantized"
print(f"Saving quantized model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Quantized model and tokenizer saved.")

# Example Input for Text Generation
prompt = "The future of AI is "
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Ensure inputs are on GPU

# Generate Text
print("Generating text with quantized model...")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

# Decode and Print Output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
