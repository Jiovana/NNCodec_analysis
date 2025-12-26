import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. LAYER-TYPE CLASSIFICATION
# =============================================================================

def classify_layer_type(layer_type):
    if layer_type.lower() in ["weight", "weights"]:
        return "weight"
    if layer_type.lower() in ["bias", "biases"]:
        return "bias"
    return "other"


# =============================================================================
# 2. SUBBLOCK CLASSIFIERS PER MODEL
# =============================================================================

# --------------------- BERT ---------------------
def classify_bert(name):
    name = name.lower()

    if "embeddings" in name or "embed" in name:
        return "embedding"

    if "layernorm" in name or "ln" in name:
        return "layernorm"

    if "self" in name and "query" in name:
        return "att_query"

    if "self" in name and "key" in name:
        return "att_key"

    if "self" in name and "value" in name:
        return "att_value"

    if "output.dense" in name:
        return "att_output"

    if "intermediate.dense" in name:
        return "mlp_fc1"

    if "output.dense" in name:
        return "mlp_fc2"

    if "pooler" in name:
        return "pooler"

    if "classifier" in name:
        return "classifier"

    return "other"


# --------------------- GPT (GPT2-like) ---------------------
def classify_gpt(name):
    name = name.lower()

    if "embed" in name:
        return "embedding"

    if "ln" in name or "layernorm" in name:
        return "layernorm"

    if "attn.c_attn" in name:
        return "attn_qkv_fused"

    if "attn.c_proj" in name:
        return "attn_output"

    if "mlp.c_fc" in name:
        return "mlp_fc1"

    if "mlp.c_proj" in name:
        return "mlp_fc2"

    return "other"


# --------------------- RESNET50 ---------------------
def classify_resnet(name):
    name = name.lower()

    if "conv" in name and "downsample" not in name:
        return "conv"

    if "bn" in name:
        return "batchnorm"

    if "downsample" in name and "0" in name:
        return "downsample_conv"

    if "downsample" in name and "1" in name:
        return "downsample_bn"

    if "fc" in name:
        return "fc"

    return "other"


# --------------------- VIT ---------------------
def classify_vit(name: str) -> str:
    """
    Classify ViT layer names into meaningful subblocks.
    """

    # --- TOKEN / EMBEDDING STAGES ---
    if "class_token" in name:
        return "class_token"
    if "pos_embedding" in name:
        return "pos_embedding"
    if "conv_proj" in name:
        return "patch_embed"

    # --- FINAL NORMALIZATION ---
    if name.startswith("encoder.ln"):
        return "final_layernorm"

    # --- CLASSIFICATION HEAD ---
    if name.startswith("heads.head"):
        return "head"

    # --- TRANSFORMER BLOCKS ---
    if "encoder.layers" in name:

        # LayerNorm 1 or 2
        if ".ln_1." in name:
            return "layernorm_1"
        if ".ln_2." in name:
            return "layernorm_2"

        # Self-attention projections
        if "self_attention.in_proj" in name:
            return "attn_in_proj"
        if "self_attention.out_proj" in name:
            return "attn_out_proj"

        # MLP structure: mlp.0 (fc1), mlp.3 (fc2)
        if ".mlp.0." in name:
            return "mlp_fc1"
        if ".mlp.3." in name:
            return "mlp_fc2"

        # If something unexpected appears inside encoder.layers
        return "transformer_other"

    # --- ANYTHING ELSE (should be rare) ---
    return "other"



# --------------------- EFFICIENTNETB0 ---------------------
def classify_efficientnet(name):
    name = name.lower()

    # stem
    if name.startswith("features.0."):
        if name.startswith("features.0.0."):
            return "stem_conv"
        if name.startswith("features.0.1."):
            return "stem_bn"
        return "stem_other"

    # match block
    m = re.search(r"features\.\d+\.\d+\.block\.(\d+)\.(.+)", name)
    if not m:
        return "other"

    subblock = int(m.group(1))
    tail = m.group(2)

    if subblock == 0:
        if tail.startswith("0."):
            return "expand_conv"
        if tail.startswith("1."):
            return "expand_bn"
        return "expand_other"

    if subblock == 1:
        if tail.startswith("0."):
            return "depthwise_conv"
        if tail.startswith("1."):
            return "depthwise_bn"
        return "depthwise_other"

    if subblock == 2:
        if tail.startswith("fc1."):
            return "se_reduce"
        if tail.startswith("fc2."):
            return "se_expand"
        return "se_other"

    if subblock == 3:
        if tail.startswith("0."):
            return "project_conv"
        if tail.startswith("1."):
            return "project_bn"
        return "project_other"

    return "other"


# =============================================================================
# 3. MASTER DISPATCHER
# =============================================================================

def classify_subblock(model, name):

    model = model.lower()

    if "bert" in model:
        return classify_bert(name)

    if "gpt" in model:
        return classify_gpt(name)

    if "resnet" in model:
        return classify_resnet(name)

    if "efficientnet" in model:
        return classify_efficientnet(name)

    if "vit" in model:
        return classify_vit(name)

    return "other"


# =============================================================================
# 4. AGGREGATION + PLOTTING
# =============================================================================

def plot_stat(df, stat, title, outdir):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[stat], kde=True)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{title.replace(' ', '_')}.png")
    plt.close()


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def process_csv(path):

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # layer type cleanup
    df["layer_type_clean"] = df["layer_type"].apply(classify_layer_type)

    # subblock classification
    df["subblock"] = df.apply(lambda r:
        classify_subblock(r["model"], r["layer_name"]), axis=1
    )

    # statistics output directories
    os.makedirs("output/stats", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)

    # aggregated stats
    group = df.groupby(["model", "layer_type_clean", "subblock"]).agg({
        "mean": "mean",
        "std": "mean",
        "skew": "mean",
        "kurtosis": "mean",
        "min": "mean",
        "max": "mean",
        "median": "mean",
        "num_params": ["mean", "sum"]
    })

    group.to_csv("output/stats/aggregated_stats.csv")

    # plotting per model / subblock
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        for sb in model_df["subblock"].unique():
            sb_df = model_df[model_df["subblock"] == sb]
            for stat in ["mean", "std", "skew", "kurtosis"]:
                title = f"{model}_{sb}_{stat}"
                plot_stat(sb_df, stat, title, "output/plots")

    print("✅ Processing complete. Results stored in output/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    process_csv("model_layer_statistics.csv")
