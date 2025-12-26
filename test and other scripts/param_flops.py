import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fvcore.nn import FlopCountAnalysis

# ---------------------------
# Load GPT-1 model and tokenizer
# ---------------------------
model_name = "openai-gpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-1 has no pad token, so we add one manually
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)
# Resize embeddings to account for the new pad token
model.resize_token_embeddings(len(tokenizer))
model.eval()

# ---------------------------
# Prepare dummy input
# ---------------------------
text = "This is a short sentence for GPT-1 FLOPs measurement."
inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=128,
    truncation=True,
    padding="max_length"
)

input_ids = inputs["input_ids"]

# ---------------------------
# FLOPs analysis
# ---------------------------
with torch.no_grad():
    flop_analyzer = FlopCountAnalysis(model, (input_ids,))
    total_flops = flop_analyzer.total()

# ---------------------------
# Parameter count
# ---------------------------
total_params = sum(p.numel() for p in model.parameters())

# ---------------------------
# Print results
# ---------------------------
print(f"Model: {model_name}")
print(f"FLOPs: {total_flops:,}")
print(f"Total parameters: {total_params:,}")
print(f"Sequence length: {input_ids.shape[1]}")
print(f"Batch size: {input_ids.shape[0]}")
