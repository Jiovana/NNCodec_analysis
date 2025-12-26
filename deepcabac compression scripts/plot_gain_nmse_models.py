import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
models = ["BERT", "GPT", "ResNet", "EfficientNet", "vit"]
input_dir = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\gain_nmse_csvs"     # directory where CSVs live
output_dir = r".\figures"
os.makedirs(output_dir, exist_ok=True)

# High-quality plotting defaults
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# Distinct colors per tensor type
color_map = {
    "weight": "#1f77b4",
    "bias": "#ff7f0e",
    "norm": "#020092",
    "norm_bias": "#8c564b",
    "buffer": "#d62728",
    "embedding": "#9467bd",
    "bn_bias": "#ff7f0e",
    "bn_weight": "#00B5A5",
    "conv_weight": "#1f77b4",
    "fc_weight": "#2ca02c",
    "fc_bias": "#ffbb78",
    "other": "#7f7f7f",
    "att_in_proj": "#e377c2",
    "att_out_proj": "#bcbd22",
    "class_token": "#17becf",
    "layernorm1": "#8c564b",
    "layernorm2": "#e377c2",
    "mlp_fc1": "#1f77b4",
    "mlp_fc2": "#ff7f0e"
}

def classify_tensor(model_name, tname):
    tname = tname.lower()

    # ---------------- BERT / GPT (generic grouping) ----------------

    if "embedding" in tname or "embed" in tname:
        return "embedding"
    if "norm" in tname or "layernorm" in tname or "bn" in tname:
        return "norm"
    if "buffer" in tname or "running" in tname:
        return "buffer"
    if "attention" in tname or "attn" in tname:
        return "attention"
    if tname.endswith("bias"):
        return "bias"
    if "mlp" in tname or "class_token" in tname or "conv" in tname or tname.endswith("weight"):
        return "weight"
    return "other"


# -----------------------------
# Main loop for each model
# -----------------------------
for model in models:
    csv_path = os.path.join(input_dir, f"summary_by_tensor_type_{model.lower()}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠ CSV not found for {model}, skipping.")
        continue

    df = pd.read_csv(csv_path)

    # Filter only 8-bit results
    df8 = df[df["bits_quant"] == 8].copy()
    if df8.empty:
        print(f"⚠ No 8-bit entries for {model}, skipping.")
        continue

    # Group tensor types
    df8["tensor_group"] = df8["tensor_type"].apply(lambda x: classify_tensor(model, x))

    groups = df8["tensor_group"].unique()

    # Colors in correct order for *this* model
    colors = [color_map.get(t, "#333333") for t in groups]

 # -------------------------
    # 1) Compression Gain plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        groups,
        [df8[df8["tensor_group"] == g]["mean_gain"].mean() for g in groups],
         yerr=[df8[df8["tensor_group"] == g]["std_gain"].mean() for g in groups],
        capsize=6,
        color=[color_map.get(t, "#333333") for t in groups],
        edgecolor="black",
        linewidth=1.3
    )

    ax.set_title(f"{model} — Mean Compression Gain (8-bit)")
    ax.set_xlabel("Tensor Type")
    ax.set_ylabel("Compression Gain (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model}_gain_8bit.svg"))
    plt.savefig(os.path.join(output_dir, f"{model}_gain_8bit.png"), dpi=400)
    plt.close()

    # -------------------------
    # 2) NMSE plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        groups,
        [df8[df8["tensor_group"] == g]["mean_nmse"].mean() for g in groups],
        color=[color_map.get(t, "#333333") for t in groups],
        edgecolor="black",
        linewidth=1.3
    )

    ax.set_title(f"{model} — Mean NMSE (8-bit)")
    ax.set_xlabel("Tensor Type")
    ax.set_ylabel("NMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model}_nmse_8bit.svg"))
    plt.savefig(os.path.join(output_dir, f"{model}_nmse_8bit.png"), dpi=400)
    plt.close()

    print(f"✓ Generated figures for {model} in {output_dir}")
