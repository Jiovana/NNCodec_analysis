import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

# ----------------------------
# CONFIG
# ----------------------------
model_name = "textattack/bert-base-uncased-SST-2"
device = "cpu"
bits_weight = 8
bits_bias = 16
bits_norm = 16

# ----------------------------
# LOAD MODEL
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ----------------------------
# UTILS
# ----------------------------
def kmeans_quantize(x, n_bits):
    """K-means quantization per layer."""
    from sklearn.cluster import KMeans


    x_flat = x.reshape(-1, 1)
    n_clusters = 2 ** n_bits
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    kmeans.fit(x_flat)
    centers = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    q = centers[labels].reshape(x.shape)
    return q.astype(np.float32)

def adaptive_quant_layer(param, layer_type):
    if layer_type == "weight":
        print("quantizing weights...")
        return kmeans_quantize(param.detach().cpu().numpy(), bits_weight)
    elif layer_type == "bias":
        print("quantizing biases...")
        return kmeans_quantize(param.detach().cpu().numpy(), bits_bias)
    elif layer_type in ["norm", "layernorm"]:
        print("quantizing norm weights...")
        return kmeans_quantize(param.detach().cpu().numpy(), bits_norm)
    else:
        return param.cpu().numpy()  # leave unchanged

# ----------------------------
# APPLY ADAPTIVE QUANTIZATION
# ----------------------------
for name, param in model.named_parameters():
    lname = name.lower()
    if "bias" in lname:
        ltype = "bias"
    elif "norm" in lname or "layernorm" in lname:
        ltype = "norm"
    elif "weight" in lname:
        ltype = "weight"
    else:
        ltype = "other"

    q_param = adaptive_quant_layer(param, ltype)
    param.data = torch.from_numpy(q_param).to(device)

# ----------------------------
# SAVE COMPRESSED MODEL
# ----------------------------
compressed_model_path = "./bert_compressed"
model.save_pretrained(compressed_model_path)
print(f"Compressed model saved to {compressed_model_path}")
