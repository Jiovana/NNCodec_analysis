import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
#  CONFIG
# -----------------------------

plt.rcParams.update({
    "font.size": 14,       # Increase base font size
    "font.weight": "bold", # Bold text everywhere
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

input_csv = r"output\stats\aggregated_stats.csv"

metrics = ["mean", "std", "skew", "kurtosis", "min", "max", "median", "num_params_sum"]
param_types = ["weight", "bias", "other"]  # ViT has "other"
color_map = {
    "mean": "tab:blue",
    "std": "tab:orange",
    "skew": "tab:green",
    "kurtosis": "tab:red",
    "min": "tab:purple",
    "max": "tab:brown",
    "median": "tab:pink",
    "num_params_sum": "tab:gray",
}

# -----------------------------
#  LOAD CSV
# -----------------------------
df = pd.read_csv(input_csv)

# Basic validation
required_cols = {"model", "param_type", "subblock_type"} | set(metrics)
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")


# -----------------------------
#  MAKE CHARTS
# -----------------------------
output_root = "charts"

for model in df["model"].unique():
    df_model = df[df["model"] == model]

    for ptype in param_types:
        df_sub = df_model[df_model["param_type"] == ptype]
        if df_sub.empty:
            continue  # skip missing param types

        for metric in metrics:
            if df_sub[metric].isna().all():
                continue

            # Create directory
            out_dir = os.path.join(output_root, model, ptype)
            os.makedirs(out_dir, exist_ok=True)

            # Plot
            plt.figure(figsize=(10, 5))
            bars = plt.bar(df_sub["subblock_type"], df_sub[metric], color=color_map[metric])
            plt.xticks(rotation=65, ha="right")
            plt.title(f"{model} – {ptype} – {metric}")
            plt.ylabel(metric)
            plt.xlabel("subblock_type")
            # --- ADD VALUE LABELS ABOVE EACH BAR ---
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.4g}",        # format numbers here
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    rotation=0               # change to rotation=45 if needed
                )
            plt.tight_layout()



            # Save file
            out_path = os.path.join(out_dir, f"{metric}.png")
            plt.savefig(out_path, dpi=200)
            plt.close()

print("All charts generated!")
