import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import sys

# ---------------------------------------------------------------------
# INPUT CSVs
# ---------------------------------------------------------------------
CSV_PARAMS = "../compression scripts/vit_b16_results_averaged.csv"

OUT_DIR = "vit_b16_quant_eval_analysis_avg"
PARAM_OUT = os.path.join(OUT_DIR, "params")

os.makedirs(PARAM_OUT, exist_ok=True)


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
        return 5
    if "conv1" in name:
        return -1
    if "bn1" in name:
        return 0
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
    if "conv_proj" in lname and "weight" in lname:
        return "conv_proj"

    # Class token
    if "class_token" in lname:
        return "class_token"

    # Positional embeddings
    if "pos_embedding" in lname:
        return "embedding"

    # LayerNorm inside blocks
    if ".ln_1." in lname or ".ln_2." in lname or lname.startswith("encoder.ln"):
        if "weight" in lname:
            return "norm_weight"
        elif "bias" in lname:
            return "norm_bias"
        else:
            return "norm"

    # Attention projection weights
    if "self_attention" in lname and "weight" in lname:
        return "attn_weight"

    # MLP blocks inside encoder layers
    if ".mlp." in lname and "weight" in lname: 
        return "mlp_weight"
    
    if "heads" in lname and "weight" in lname:
        return "class_weight"
    
    if "bias" in lname:
        return "bias"
        
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
    m = re.search(r"encoder\.layers\.encoder_layer_(\d+)", name)
    if m:
        return int(m.group(1))
    else:
        return -1


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
        mean_enc_time=("encode_time", "mean"),
        mean_dec_time=("decode_time", "mean"),
        mean_peak_eram=("peak_enc_mem", "mean"),
        mean_peak_dram=("peak_dec_mem", "mean"),
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
# MODE-BASED STATISTICS
# ---------------------------------------------------------------------

# Summary per tensor_type and mode_used
summary_type_mode = (
    df_p.groupby(["bits_quant", "tensor_type", "mode_used"])
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        std_ratio=("compression_ratio", "std"),
        mean_gain=("compression_gain_pct", "mean"),
        std_gain=("compression_gain_pct", "std"),
        mean_bpe=("bits_per_element", "mean"),
        std_bpe=("bits_per_element", "std"),
        mean_nmse=("nmse_post", "mean"),
        std_nmse=("nmse_post", "std"),
        mean_enc_time=("encode_time", "mean"),
        mean_dec_time=("decode_time", "mean"),
        mean_peak_eram=("peak_enc_mem", "mean"),
        mean_peak_dram=("peak_dec_mem", "mean"),
        total_tensors=("numel", "count")
    )
    .reset_index()
)

# % tensors in bypass mode
bypass_frac = (
    df_p[df_p["mode_used"] == "bypass"]
    .groupby(["bits_quant", "tensor_type"])
    .size()
    / df_p.groupby(["bits_quant", "tensor_type"]).size()
).reset_index(name="bypass_frac")
bypass_frac["bypass_frac_pct"] = bypass_frac["bypass_frac"] * 100

summary_type_mode = summary_type_mode.merge(
    bypass_frac, on=["bits_quant", "tensor_type"], how="left"
)

# Pivot for differences
pivot_mode = summary_type_mode.pivot_table(
    index=["bits_quant", "tensor_type"],
    columns="mode_used",
    values=["mean_enc_time", "mean_peak_eram", "mean_dec_time", "mean_peak_dram"]
).reset_index()

pivot_mode["delta_enc_time"] = pivot_mode["mean_enc_time"]["regular"] - pivot_mode["mean_enc_time"]["bypass"]
pivot_mode["delta_dec_time"] = pivot_mode["mean_dec_time"]["regular"] - pivot_mode["mean_dec_time"]["bypass"]
pivot_mode["delta_peak_eram"] = pivot_mode["mean_peak_eram"]["regular"] - pivot_mode["mean_peak_eram"]["bypass"]
pivot_mode["delta_peak_dram"] = pivot_mode["mean_peak_dram"]["regular"] - pivot_mode["mean_peak_dram"]["bypass"]


# ---------------------------------------------------------------------
# SAVE PARAM SUMMARIES
# ---------------------------------------------------------------------
summary_type_p.to_csv(os.path.join(PARAM_OUT, "summary_by_tensor_type.csv"), index=False)
summary_layer_p.to_csv(os.path.join(PARAM_OUT, "summary_by_layer.csv"), index=False)
summary_overall_p.to_csv(os.path.join(PARAM_OUT, "summary_overall.csv"), index=False)
summary_global_p.to_csv(os.path.join(PARAM_OUT, "overall_compression_summary.csv"), index=False)
summary_type_mode.to_csv(os.path.join(PARAM_OUT, "summary_by_tensor_type_and_mode.csv"), index=False)
pivot_mode.to_csv(os.path.join(PARAM_OUT, "mode_comparison.csv"), index=False)


# === Plotting ===
for bits in sorted(df_p["bits_quant"].unique()):
    # --- Compression Gain by Tensor Type ---
    sub_type = summary_type_p[summary_type_p["bits_quant"] == bits]
    plt.figure(figsize=(7, 4))
    plt.bar(sub_type["tensor_type"], sub_type["mean_gain"], yerr=sub_type["std_gain"], capsize=3, color="#7A5FC0")
    plt.ylabel("Compression Gain (%)")
    plt.title(f"Mean Compression Gain by Tensor Type (q={bits}-bit)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"gain_by_type_q{bits}.png"))
    plt.close()

    # --- NMSE by Tensor Type ---
    plt.figure(figsize=(7, 4))
    plt.bar(sub_type["tensor_type"], sub_type["mean_nmse"], yerr=sub_type["std_nmse"], capsize=3, color="orange")
    plt.ylabel("NMSE")
    plt.title(f"Normalized MSE by Tensor Type (q={bits}-bit)")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"nmse_by_type_q{bits}.png"))
    plt.close()

    # --- Compression Gain per Layer (embeddings + transformer blocks) ---
    sub_layer = summary_layer_p[summary_layer_p["bits_quant"] == bits]
    plt.figure(figsize=(7, 4))
    plt.plot(sub_layer["layer_id"], sub_layer["mean_gain"], marker="o", color="#7A5FC0")
    plt.xlabel("Layer ID ")
    plt.ylabel("Compression Gain (%)")
    plt.title(f"Compression Gain per Layer (q={bits}-bit)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"gain_by_layer_q{bits}.png"))
    plt.close()

    # --- NMSE per Layer ---
    plt.figure(figsize=(7, 4))
    plt.plot(sub_layer["layer_id"], sub_layer["mean_nmse"], marker="o", color="orange")
    plt.xlabel("Layer ID ")
    plt.ylabel("NMSE")
    plt.title(f"NMSE per Layer (q={bits}-bit)")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"nmse_by_layer_q{bits}.png"))
    plt.close()

    # --- Scatter NMSE vs Compression Gain ---
    sub = df_p[df_p["bits_quant"] == bits]
    plt.figure(figsize=(7, 5))
    plt.scatter(sub["compression_gain_pct"], sub["nmse_post"], alpha=0.6, edgecolor='k', s=30, color="#7A5FC0")
    plt.xlim(left=0)
    plt.yscale("log")
    plt.xlabel("Compression Gain (%)")
    plt.ylabel("NMSE")
    plt.title(f"NMSE vs Compression Gain (q={bits}-bit)")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"nmse_vs_gain_q{bits}.png"))
    plt.close()

    # === RAM HISTOGRAMS (ENC / DEC) ===
    sub_df = df_p[df_p["bits_quant"] == bits]

    def plot_ram_hist(series, title, xlabel, out_name, log=False):
        plt.figure(figsize=(7, 4))

        counts, bins, patches = plt.hist(
            series,
            bins=40,
            edgecolor="black",
            alpha=0.75, color="#7A5FC0"
        )

        if log:
            plt.xscale("log")

        # Add labels on top of each bar
        for count, patch in zip(counts, patches):
            if count >= counts.max() * 0.1:
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                plt.text(
                    x, y,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0
                )

        plt.xlabel(xlabel)
        plt.ylabel("Number of Tensors")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, out_name))
        plt.close()

    # Peak RSS (absolute)
    plot_ram_hist(
        sub_df["peak_enc_mem"],
        f"Peak Encoder RAM Usage ({bits}-bit)",
        "Peak RAM [bytes]",
        f"hist_peak_enc_ram_{bits}.png"
    )

    plot_ram_hist(
        sub_df["peak_dec_mem"],
        f"Peak Decoder RAM Usage ({bits}-bit)",
        "Peak RAM [bytes]",
        f"hist_peak_dec_ram_{bits}.png"
    )

    # Delta RAM (incremental)
    plot_ram_hist(
        sub_df["peak_delta_enc"],
        f"Incremental Encoder RAM (ΔRSS, {bits}-bit)",
        "ΔRAM [bytes]",
        f"hist_delta_enc_ram_{bits}.png"
    )

    plot_ram_hist(
        sub_df["peak_delta_dec"],
        f"Incremental Decoder RAM (ΔRSS, {bits}-bit)",
        "ΔRAM [bytes]",
        f"hist_delta_dec_ram_{bits}.png"
    )

    def summarize_ram(series, name):
        return {
            f"{name}_median": np.median(series),
            f"{name}_p95": np.percentile(series, 95),
            f"{name}_max": np.max(series),
        }


    sub_df = df_p[df_p["bits_quant"] == bits]
    ram_summary = {}
    ram_summary.update(summarize_ram(sub_df["peak_enc_mem"], "peak_enc_mem"))
    ram_summary.update(summarize_ram(sub_df["peak_dec_mem"], "peak_dec_mem"))
    ram_summary.update(summarize_ram(sub_df["peak_delta_enc"], "peak_delta_enc"))
    ram_summary.update(summarize_ram(sub_df["peak_delta_dec"], "peak_delta_dec"))

    ram_summary_df = pd.DataFrame([ram_summary])
    ram_summary_df.to_csv(
        os.path.join(OUT_DIR, f"ram_summary_stats_q{bits}.csv"),
        index=False
    )

    print(f"\n=== RAM Summary Statistics ( {bits}-bit) ===")
    print(ram_summary_df.T)






# ---------------------------------------------------------------------
print("\n=== PARAM Summaries ===")
print(summary_overall_p)
print(summary_global_p)



print("\n=== DONE. Files saved in ===")
print(" →", PARAM_OUT)
