import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import sys

# ---------------------------------------------------------------------
# INPUT CSVs
# ---------------------------------------------------------------------
CSV_PARAMS = "../multi_model_quant_eval/efficientnet_b0_compression.csv"
CSV_BUFFERS = "../multi_model_quant_eval/efficientnet_b0_buffers_compression.csv"

OUT_DIR = "efficientnet_b0_quant_eval_analysis"
PARAM_OUT = os.path.join(OUT_DIR, "params")
BUFFER_OUT = os.path.join(OUT_DIR, "buffers")
os.makedirs(PARAM_OUT, exist_ok=True)
os.makedirs(BUFFER_OUT, exist_ok=True)

# Detect model type from params filename
csv_name = os.path.basename(CSV_PARAMS).lower()
if "resnet" in csv_name:
    MODEL_TYPE = "resnet"
elif "efficientnet" in csv_name:
    MODEL_TYPE = "efficientnet"
elif "vit" in csv_name:
    MODEL_TYPE = "vit"
else:
    print("❌ Could not detect model type from filename!")
    sys.exit(1)

print(f"➡ Using MODEL_TYPE = {MODEL_TYPE}")

# ---------------------------------------------------------------------
# LOAD CSVs
# ---------------------------------------------------------------------
df_p = pd.read_csv(CSV_PARAMS)
df_p = df_p[df_p["compressed_bytes"] > 0]

df_b = pd.read_csv(CSV_BUFFERS)
df_b = df_b[df_b["compressed_bytes"] > 0]


# =============================================================================
# PARAM CLASSIFICATION (ONLY FOR PARAMS)
# =============================================================================

def classify_param_resnet(name):
    lname = name.lower()
    if "conv" in lname:
        return "conv_weight" if "weight" in lname else "conv_bias"
    if "bn" in lname:
        return "bn_weight" if "weight" in lname else "bn_bias"
    if "fc" in lname:
        return "fc_weight" if "weight" in lname else "fc_bias"
    if "downsample" and "0" in lname:
        return "downsample_conv_weight" if "weight" in lname else "downsample_conv_bias"
    if "downsample" and "1" in lname:
        return "downsample_bn_weight" if "weight" in lname else "downsample_bn_bias"
    return "other"

def extract_layer_resnet(name):
    m = re.search(r"layer(\d+)", name)
    if m:
        return int(m.group(1))
    if "fc" in name:
        return 100
    return -1

def classify_param_efficientnet(name):
    lname = name.lower()

    # -------------------------
    # Stem
    # -------------------------
    if lname.startswith("features.0."):
        if ".0.weight" in lname:
            return "stem_conv"
        if ".1.weight" in lname or ".1.bias" in lname:
            return "batch_norm"
        return "other"

    # -------------------------
    # Biases (grouped)
    # -------------------------
    if lname.endswith(".bias"):
        return "bias"
    
    # -------------------------
    # Classifier
    # -------------------------
    if lname.startswith("classifier"):
        return "classifier"

    # -------------------------
    # SE fully-connected layers
    # -------------------------
    if ".fc1." in lname or ".fc2." in lname:
        return "se_fc"

    # -------------------------
    # BatchNorm (all blocks)
    # -------------------------
    # BN is always index ".1." in EfficientNet
    if ".1.weight" in lname or ".1.bias" in lname:
        return "batch_norm"

    # -------------------------
    # Convolutions
    # -------------------------
    # All conv weights end in ".weight" and are ".0.weight"
    if lname.endswith(".weight") and ".0.weight" in lname:
        return "conv"  # refined later using shape

    

    return "other"

def extract_layer_efficientnet(name):
    m = re.search(r"features\.(\d+)", name)
    return int(m.group(1)) if m else -1

def classify_param_vit(name):
    lname = name.lower()

    # Patch embedding projection (Conv on input patches)
    if "patch_embed" in lname or "conv_proj" in lname:
        if "weight" in lname:
            return "conv_proj"

    # Class token
    elif "class_token" in lname:
        return "class_token"

    # Positional embeddings
    elif "pos_embedding" in lname:
        return "embedding"

    # LayerNorm inside blocks
    elif "ln_1" in lname or "ln_2" in lname or "ln" in lname:
        if "weight" in lname:
            return "norm_weight"
        elif "bias" in lname:
            return "norm_bias"
        else:
            return "norm"

    # Attention projection weights
    elif "self_attention" in lname and "weight" in lname:
        return "attn_weight"

    # MLP blocks inside encoder layers
    elif "mlp" in lname and "weight" in lname: 
        return "mlp_weight"
    
    elif "heads" in lname and "weight" in lname:
        return "class_weight"
    
    elif "bias" in lname:
        return "bias"
        
    else:
        return "other"

def refine_efficientnet_conv(row):
    if row["tensor_type"] != "conv":
        return row["tensor_type"]

    shape = row.get("shape", None)
    if not isinstance(shape, str):
        return "pw_conv"

    try:
        dims = eval(shape)
    except Exception:
        return "pw_conv"

    # Depthwise conv: [C, 1, k, k]
    if len(dims) == 4 and dims[1] == 1:
        return "dw_conv"

    # Pointwise conv: [C_out, C_in, 1, 1]
    if len(dims) == 4 and dims[2] == 1 and dims[3] == 1:
        return "pw_conv"

    return "pw_conv"


def extract_layer_vit(name):
    """
    Extract encoder layer index:
    encoder.layers.<i>.xxx
    """
    m = re.search(r"encoder\.layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def classify_param(name):
    if MODEL_TYPE == "resnet":
        return classify_param_resnet(name)
    if MODEL_TYPE == "efficientnet":
        return classify_param_efficientnet(name)
    if MODEL_TYPE == "vit":
        return classify_param_vit(name)
    return "other"

def extract_layer(name):
    if MODEL_TYPE == "resnet":
        return extract_layer_resnet(name)
    if MODEL_TYPE == "efficientnet":
        return extract_layer_efficientnet(name)
    if MODEL_TYPE == "vit":
        return extract_layer_vit(name)
    return -1


# ---------------------------------------------------------------------
# APPLY CLASSIFICATION FOR PARAMS ONLY
# ---------------------------------------------------------------------
df_p["tensor_type"] = df_p["param_name"].apply(classify_param)
if MODEL_TYPE == "efficientnet":
    df_p["tensor_type"] = df_p.apply(refine_efficientnet_conv, axis=1)
df_p["layer_id"] = df_p["param_name"].apply(extract_layer)

# Compute gain for params if missing
if "compression_gain_pct" not in df_p.columns:
    df_p["compression_gain_pct"] = (1 - 1 / df_p["compression_ratio"]) * 100


# ---------------------------------------------------------------------
# PARAM SUMMARY STATISTICS
# ---------------------------------------------------------------------
summary_type_p = (
    df_p.groupby(["bits_quant", "tensor_type"])
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        std_ratio=("compression_ratio", "std"),
        mean_gain=("compression_gain_pct", "mean"),
        std_gain=("compression_gain_pct", "std"),
        mean_bpe=("bits_per_element", "mean"),
        std_bpe=("bits_per_element", "std"),
        mean_nmse=("nmse_post", "mean"),
        std_nmse=("nmse_post", "std"),
        raw_total_bytes=("raw_bytes", "sum"),
        comp_total_bytes=("compressed_bytes", "sum"),
        total_elems=("numel", "sum"),
    )
    .reset_index()
)

# Add overhead metrics
summary_type_p["overhead_bytes"] = (
    summary_type_p["comp_total_bytes"] - summary_type_p["raw_total_bytes"]
)
summary_type_p["overhead_pct"] = (
    summary_type_p["overhead_bytes"] / summary_type_p["raw_total_bytes"] * 100
).replace([np.inf, -np.inf], np.nan)

summary_layer_p = (
    df_p.groupby(["bits_quant", "layer_id"])
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        mean_gain=("compression_gain_pct", "mean"),
        mean_bpe=("bits_per_element", "mean"),
        mean_nmse=("nmse_post", "mean"),
        raw_total_bytes=("raw_bytes", "sum"),
        comp_total_bytes=("compressed_bytes", "sum"),
    )
    .reset_index()
    .sort_values("layer_id")
)

# Add overhead
summary_layer_p["overhead_bytes"] = (
    summary_layer_p["comp_total_bytes"] - summary_layer_p["raw_total_bytes"]
)
summary_layer_p["overhead_pct"] = (
    summary_layer_p["overhead_bytes"] / summary_layer_p["raw_total_bytes"] * 100
).replace([np.inf, -np.inf], np.nan)

summary_overall_p = (
    df_p.groupby("bits_quant")
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        std_ratio=("compression_ratio", "std"),
        mean_gain=("compression_gain_pct", "mean"),
        std_gain=("compression_gain_pct", "std"),
        mean_nmse=("nmse_post", "mean"),
        std_nmse=("nmse_post", "std"),
        raw_total_bytes=("raw_bytes", "sum"),
        comp_total_bytes=("compressed_bytes", "sum"),
    )
    .reset_index()
)

# Add overhead
summary_overall_p["overhead_bytes"] = (
    summary_overall_p["comp_total_bytes"] - summary_overall_p["raw_total_bytes"]
)
summary_overall_p["overhead_pct"] = (
    summary_overall_p["overhead_bytes"] / summary_overall_p["raw_total_bytes"] * 100
).replace([np.inf, -np.inf], np.nan)

summary_global_p = []
for bits, sub in df_p.groupby("bits_quant"):
    raw_total = sub["raw_bytes"].sum()
    comp_total = sub["compressed_bytes"].sum()
    ratio_total = raw_total / comp_total
    gain_total = (1 - 1 / ratio_total) * 100
    summary_global_p.append({
        "bits_quant": bits,
        "raw_total_bytes": raw_total,
        "compressed_total_bytes": comp_total,
        "overall_ratio": ratio_total,
        "overall_gain_pct": gain_total,
        "overall_overhead_bytes": int(comp_total - raw_total),
        "overall_overhead_pct": float((comp_total - raw_total) / raw_total * 100) if raw_total > 0 else "",
    })
summary_global_p = pd.DataFrame(summary_global_p)


# ---------------------------------------------------------------------
# SAVE PARAM SUMMARIES
# ---------------------------------------------------------------------
summary_type_p.to_csv(os.path.join(PARAM_OUT, "summary_by_tensor_type.csv"), index=False)
summary_layer_p.to_csv(os.path.join(PARAM_OUT, "summary_by_layer.csv"), index=False)
summary_overall_p.to_csv(os.path.join(PARAM_OUT, "summary_overall.csv"), index=False)
summary_global_p.to_csv(os.path.join(PARAM_OUT, "overall_compression_summary.csv"), index=False)

# ---------------------------------------------------------------------
# BUFFER SUMMARY (INDEPENDENT)
# ---------------------------------------------------------------------
summary_buffer = (
    df_b.groupby("bits_quant")
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        std_ratio=("compression_ratio", "std"),
        mean_gain=("compression_gain_pct", "mean"),
        std_gain=("compression_gain_pct", "std"),
        mean_nmse=("nmse_post", "mean"),
        std_nmse=("nmse_post", "std"),
        total_elems=("numel", "sum"),
    )
    .reset_index()
)

# per-bits global totals (raw bytes, compressed bytes, overall ratio/gain)
summary_global_b = []
for bits, sub in df_b.groupby("bits_quant"):
    raw_total = sub["raw_bytes"].sum()
    comp_total = sub["compressed_bytes"].sum()
    # guard against zero compressed bytes
    if comp_total == 0 or np.isnan(comp_total):
        ratio_total = np.nan
        gain_total = np.nan
    else:
        ratio_total = raw_total / comp_total
        gain_total = (1 - 1 / ratio_total) * 100
    summary_global_b.append({
        "bits_quant": int(bits),
        "raw_total_bytes": int(raw_total),
        "compressed_total_bytes": int(comp_total),
        "overall_ratio": float(ratio_total) if not np.isnan(ratio_total) else "",
        "overall_gain_pct": float(gain_total) if not np.isnan(gain_total) else "",
        "overall_overhead_bytes": int(comp_total - raw_total),
        "overall_overhead_pct": float((comp_total - raw_total) / raw_total * 100) if raw_total > 0 else "",
    })
summary_global_b = pd.DataFrame(summary_global_b)


# Save buffer summaries
summary_buffer.to_csv(os.path.join(BUFFER_OUT, "summary_buffers.csv"), index=False)
summary_global_b.to_csv(os.path.join(BUFFER_OUT, "overall_compression_summary.csv"), index=False)


# ---------------------------------------------------------------------
# BUFFER PLOTS
# ---------------------------------------------------------------------
for bits in df_b["bits_quant"].unique():
    sub = df_b[df_b["bits_quant"] == bits]

    plt.figure(figsize=(7, 4))
    plt.hist(sub["nmse_post"], bins=40)
    plt.yscale("log")
    plt.title(f"Buffers NMSE Distribution ({bits}-bit)")
    plt.xlabel("NMSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(BUFFER_OUT, f"buffer_nmse_hist_q{bits}.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(sub["compression_ratio"], bins=40)
    plt.title(f"Buffers Compression Ratio Distribution ({bits}-bit)")
    plt.xlabel("Compression Ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(BUFFER_OUT, f"buffer_ratio_hist_q{bits}.png"))
    plt.close()

# ---------------------------------------------------------------------
print("\n=== PARAM Summaries ===")
print(summary_overall_p)
print(summary_global_p)

print("\n=== BUFFER Summaries ===")
print(summary_buffer)
print(summary_global_b)

print("\n=== DONE. Files saved in ===")
print(" →", PARAM_OUT)
print(" →", BUFFER_OUT)
