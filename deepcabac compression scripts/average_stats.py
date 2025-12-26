import pandas as pd
from pathlib import Path

# collect all run CSVs
csvs = sorted(Path(".").glob("multi_model_quant_eval_run*/resnet50_compression.csv"))
#csvs = sorted(Path(".").glob("gpt_quant_eval_mixed_run*/compression_results.csv"))

dfs = []
for f in csvs:
    df = pd.read_csv(f)
    df["run"] = f.parent.name
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# columns to average
avg_cols = [
    "encode_time", "decode_time",
    "peak_enc_mem", "peak_dec_mem",
    "peak_delta_enc", "peak_delta_dec"
]
#gpt columns param_name,layer_id,subblock,layer_type,shape,numel,bits_quant,raw_bytes,qstep,mse_quant,
# nmse_quant,mse_post,nmse_post,compressed_bytes,compression_ratio,compression_gain_pct,
# bits_per_element,decoded_equal,peak_enc_mem,peak_dec_mem,encode_time,decode_time,
# peak_delta_enc,peak_delta_dec


# columns to keep (take first, they should match)
""" id_cols = [ berto columns
    "param_name", "layer_type", "shape", "numel",
    "bits_quant", "raw_bytes", "qstep",
    "mse_quant", "nmse_quant",
    "mse_post", "nmse_post",
    "compressed_bytes", "compression_ratio",
    "compression_gain_pct", "bits_per_element",
    "decoded_equal"
] """

"""id_cols = [ GPT columns
    "param_name", "layer_id", "subblock", "layer_type", "shape", "numel",
    "bits_quant", "raw_bytes", "qstep",
    "mse_quant", "nmse_quant","mse_post", "nmse_post",
    "compressed_bytes", "compression_ratio",
    "compression_gain_pct", "bits_per_element",
    "decoded_equal"
]"""

# vision columns
id_cols = [ 
    "param_name", "layer_type", "shape", "numel",
    "bits_quant", "raw_bytes", "qstep",
    "mse_quant", "nmse_quant",
    "mse_post", "nmse_post",
    "compressed_bytes", "compression_ratio",
    "compression_gain_pct", "bits_per_element",
    "decoded_equal", "mode_used"
]

agg_dict = {c: "mean" for c in avg_cols}
agg_dict.update({c: "first" for c in id_cols})

avg_df = (
    all_df
    .groupby(["param_name", "bits_quant"], as_index=False)
    .agg(agg_dict)
)
avg_df.to_csv("resnet50_results_averaged.csv", index=False)
