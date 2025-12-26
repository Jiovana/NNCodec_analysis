import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os

# === Paths ===
CSV_PATH = "../compression scripts/gpt_results_averaged.csv"
OUT_DIR = "gpt_quant_eval_mixed_analysis_avg"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load ===
df = pd.read_csv(CSV_PATH, header=0)
df = df[df["compressed_bytes"] > 0]  # drop invalid rows

print(df.columns)
print(df.head())

# --- classify tensors using 'shape' and 'numel' ---
def classify_tensor(row):
    # shape column tells us the layer type: embedding, attention, layernorm, mlp
    # numel column tells us weight vs bias
    subblock = row['subblock']
    layer_type = row['layer_type']

    if subblock == 'embedding':
        return 'embedding'
    elif subblock == 'layernorm':
        if 'weight' in layer_type:
            return 'norm_weight'
        elif 'bias' in layer_type:
            return 'norm_bias'
        else:
            return 'norm'
    elif subblock in ['attention', 'mlp']:
        if 'weight' in layer_type:
            return 'weight'
        elif 'bias' in layer_type:
            return 'bias'
        else:
            return 'unknown'
    else:
        return 'unknown'

df['tensor_type'] = df.apply(classify_tensor, axis=1)

# === Derived metrics ===
if "compression_gain_pct" not in df.columns:
    df["compression_gain_pct"] = (1 - 1 / df["compression_ratio"].astype(float)) * 100

# === Per tensor type summary ===
summary_type = (
    df.groupby(["bits_quant", "tensor_type"])
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        std_ratio=("compression_ratio", "std"),
        mean_gain=("compression_gain_pct", "mean"),
        std_gain=("compression_gain_pct", "std"),
        mean_bpe=("bits_per_element", "mean"),
        std_bpe=("bits_per_element", "std"),
        mean_nmse=("nmse_post", "mean"),
        std_nmse=("nmse_post", "std"),
        total_elems=("numel", "sum"),
    )
    .reset_index()
    .sort_values(["bits_quant", "tensor_type"])
)

# === Per layer summary (including embeddings) ===
summary_layer = (
    df.groupby(["bits_quant", "layer_id"])
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        mean_gain=("compression_gain_pct", "mean"),
        mean_bpe=("bits_per_element", "mean"),
        mean_nmse=("nmse_post", "mean"),
    )
    .reset_index()
    .sort_values("layer_id")
)

# === Overall summary ===
overall = (
    df.groupby("bits_quant")
    .agg(
        mean_ratio=("compression_ratio", "mean"),
        std_ratio=("compression_ratio", "std"),
        mean_gain=("compression_gain_pct", "mean"),
        std_gain=("compression_gain_pct", "std"),
        mean_nmse=("nmse_post", "mean"),
        std_nmse=("nmse_post", "std"),
    )
    .reset_index()
)

# === Global weighted summary ===
summary_global = []
for bits, sub in df.groupby("bits_quant"):
    raw_total = sub["raw_bytes"].sum()
    comp_total = sub["compressed_bytes"].sum()
    ratio_total = raw_total / comp_total
    gain_total = (1 - 1 / ratio_total) * 100
    summary_global.append({
        "bits_quant": bits,
        "raw_total_bytes": raw_total,
        "compressed_total_bytes": comp_total,
        "overall_ratio": ratio_total,
        "overall_gain_pct": gain_total,
    })
summary_global = pd.DataFrame(summary_global)

# === Save summaries ===
summary_type.to_csv(os.path.join(OUT_DIR, "summary_by_tensor_type.csv"), index=False)
summary_layer.to_csv(os.path.join(OUT_DIR, "summary_by_layer.csv"), index=False)
overall.to_csv(os.path.join(OUT_DIR, "summary_overall.csv"), index=False)
summary_global.to_csv(os.path.join(OUT_DIR, "overall_compression_summary.csv"), index=False)

# === Plotting ===
for bits in sorted(df["bits_quant"].unique()):
    # --- Compression Gain by Tensor Type ---
    sub_type = summary_type[summary_type["bits_quant"] == bits]
    plt.figure(figsize=(7, 4))
    plt.bar(sub_type["tensor_type"], sub_type["mean_gain"], yerr=sub_type["std_gain"], capsize=3, color="#E273A9")
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
    sub_layer = summary_layer[summary_layer["bits_quant"] == bits]
    plt.figure(figsize=(7, 4))
    plt.plot(sub_layer["layer_id"], sub_layer["mean_gain"], marker="o", color="#E273A9")
    plt.xlabel("Layer ID (-1 = embeddings)")
    plt.ylabel("Compression Gain (%)")
    plt.title(f"Compression Gain per Layer (q={bits}-bit)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"gain_by_layer_q{bits}.png"))
    plt.close()

    # --- NMSE per Layer ---
    plt.figure(figsize=(7, 4))
    plt.plot(sub_layer["layer_id"], sub_layer["mean_nmse"], marker="o", color="orange")
    plt.xlabel("Layer ID (-1 = embeddings)")
    plt.ylabel("NMSE")
    plt.title(f"NMSE per Layer (q={bits}-bit)")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"nmse_by_layer_q{bits}.png"))
    plt.close()

    # --- Scatter NMSE vs Compression Gain ---
    sub = df[df["bits_quant"] == bits]
    plt.figure(figsize=(7, 5))
    plt.scatter(sub["compression_gain_pct"], sub["nmse_post"], alpha=0.6, edgecolor='k', s=30, color="#E273A9")
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
    sub_df = df[df["bits_quant"] == bits]

    def plot_ram_hist(series, title, xlabel, out_name, log=False):
        plt.figure(figsize=(7, 4))

        counts, bins, patches = plt.hist(
            series,
            bins=40,
            edgecolor="black",
            alpha=0.75, color="#E273A9"
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
        f"Peak Encoder RAM Usage (BERT, {bits}-bit)",
        "Peak RAM [bytes]",
        f"hist_peak_enc_ram_{bits}.png"
    )

    plot_ram_hist(
        sub_df["peak_dec_mem"],
        f"Peak Decoder RAM Usage (BERT, {bits}-bit)",
        "Peak RAM [bytes]",
        f"hist_peak_dec_ram_{bits}.png"
    )

    # Delta RAM (incremental)
    plot_ram_hist(
        sub_df["peak_delta_enc"],
        f"Incremental Encoder RAM (ΔRSS, BERT, {bits}-bit)",
        "ΔRAM [bytes]",
        f"hist_delta_enc_ram_{bits}.png"
    )

    plot_ram_hist(
        sub_df["peak_delta_dec"],
        f"Incremental Decoder RAM (ΔRSS, BERT, {bits}-bit)",
        "ΔRAM [bytes]",
        f"hist_delta_dec_ram_{bits}.png"
    )

    def summarize_ram(series, name):
        return {
            f"{name}_median": np.median(series),
            f"{name}_p95": np.percentile(series, 95),
            f"{name}_max": np.max(series),
        }


    sub_df = df[df["bits_quant"] == bits]
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

    print(f"\n=== RAM Summary Statistics (BERT, {bits}-bit) ===")
    print(ram_summary_df.T)

# === Print summaries ===
print("=== Global Summary (Total Bytes) ===")
print(summary_global)
print("\n=== Overall Summary (Mean/Std) ===")
print(overall)
print("\n=== Per Tensor Type ===")
print(summary_type)
print("\n=== Per Layer (Transformer + Embeddings) ===")
print(summary_layer)
print(f"\nResults saved in: {OUT_DIR}")
