import torch
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from torchvision import models

# ---------------- Discrete entropy function ----------------
def discrete_shannon_entropy(arr, num_bits=8):
    qmin = -(2**(num_bits-1))
    qmax = 2**(num_bits-1) - 1
    q_arr = np.round(np.clip(arr, qmin, qmax)).astype(np.int32)
    vals, counts = np.unique(q_arr, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

# ---------------- Layer statistics ----------------
def layer_statistics(name, param):
    data = param.detach().cpu().numpy().astype(np.float32)
    
    # heuristically classify layer type
    lname = name.lower()
    if "weight" in lname:
        layer_type = "weight"
    elif "bias" in lname:
        layer_type = "bias"
    elif "norm" in lname or "layernorm" in lname:
        layer_type = "norm"
    else:
        layer_type = "other"
    
    stats = {
        "layer_name": name,
        "layer_type": layer_type,
        "shape": data.shape,
        "num_params": data.size,
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "skew": float(skew(data.flatten())),
        "kurtosis": float(kurtosis(data.flatten())),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "entropy_8bit": float(discrete_shannon_entropy(data, num_bits=8)),
        "entropy_16bit": float(discrete_shannon_entropy(data, num_bits=16))
    }
    return stats

# ---------------- Model list ----------------
models_to_analyze = {
    "BERT": lambda: AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2"),
    "GPT": lambda: AutoModelForCausalLM.from_pretrained("openai-gpt"),
    "ResNet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    "EfficientNetB0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "ViT": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
}

all_stats = []

for model_name, model_fn in models_to_analyze.items():
    print(f"Processing {model_name}...")
    model = model_fn()
    model.eval()
    
    for name, param in model.named_parameters():
        stats = layer_statistics(name, param)
        stats["model"] = model_name
        all_stats.append(stats)

# ---------------- Save results ----------------
df = pd.DataFrame(all_stats)
df.to_csv("model_layer_statistics.csv", index=False)
print("Statistics saved to model_layer_statistics.csv")
