# bert_deepcabac_quant_eval_mixed.py
import os
import csv
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from nncodec.extensions import deepCABAC
import psutil, time, threading
from timeit import default_timer as timer

OUT_DIR = "bert_quant_eval_mixed"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "textattack/bert-base-uncased-SST-2"

# HDSP defaults for encodeLayer
CHAN_SKIP_DEFAULT = lambda num_ch: np.zeros(num_ch if num_ch>0 else 1, dtype=np.int32)
HDSP_MODE = deepCABAC.HdspMode.AlwaysOff
HDSP_HIST = np.zeros(256, dtype=np.int8)

# Encoder context model defaults (tweak if desired)
CABAC_UNARY_LEN = 4
PARAM_OPT_FLAG = 1

# === QUANT LAYER DEFAULT ARGUMENTS PER LAYER TYPE ===
QUANTLAYER_ARGS = {
    "weight": dict(
        dq_flag=0,          # URQ
        qpDensity=2,
        qp=-32,
        lambdaScale=0.0,
        maxNumNoRem=10,
        scan_order=0,
        general_profile_idc=0
    ),

    # these values are placeholders; uniform-optimizer will override qstep anyway
    "bias": dict(
        dq_flag=1,
        qpDensity=2,
        qp=-50,
        lambdaScale=0.05,
        maxNumNoRem=0,
        scan_order=0,
        general_profile_idc=0
    ),

    "norm": dict(
        dq_flag=1,
        qpDensity=2,
        qp=-50,
        lambdaScale=0.05,
        maxNumNoRem=0,
        scan_order=0,
        general_profile_idc=0
    ),

    "other": dict(
        dq_flag=1,
        qpDensity=2,
        qp=-32,
        lambdaScale=0.0,
        maxNumNoRem=0,
        scan_order=0,
        general_profile_idc=0
    ),
}

# ---------------- utilities ----------------
def compute_qstep(qp, qpDensity):
    """Replicate the QP -> qStep mapping used in quantLayer."""
    k = 1 << qpDensity
    mul = k + (qp & (k - 1))
    shift = int(qp) >> qpDensity
    qstep = float(mul) * (2.0 ** (shift - qpDensity))
    return qstep

def convert_bitdepth(q_int32, bitwidth, unsigned=False):
    """Clip quantized integers to target bitwidth (signed/unsigned)."""
    if unsigned:
        qmin = 0
        qmax = (1 << bitwidth) - 1
    else:
        qmin = -(1 << (bitwidth - 1))
        qmax = (1 << (bitwidth - 1)) - 1
    return np.clip(q_int32, qmin, qmax).astype(np.int32)

# ---------------- optimized uniform quantizer ----------------
def optimal_uniform_quant(x, bitwidth, search_steps=40):
    """
    Fast per-tensor MSE-optimal symmetric uniform quantization.
    Returns (q_int32, qstep).
    """
    x = x.astype(np.float32)
    Qmax = (1 << (bitwidth - 1)) - 1

    # trivial case
    if x.size == 0 or np.all(x == 0):
        return np.zeros_like(x, dtype=np.int32), 1.0

    std = float(np.std(x))
    # if std is tiny, still need small qstep; use dynamic bounds
    if std == 0:
        qstep_opt = 1.0
        q = np.zeros_like(x, dtype=np.int32)
        return q, qstep_opt

    # initial search interval (empirically robust)
    qstep_min = max(std / (1 << (bitwidth + 2)), 1e-12)
    qstep_max = max(std * 4.0, qstep_min * 2.0)

    phi = (1 + np.sqrt(5)) / 2.0
    invphi = 1.0 / phi

    a, b = qstep_min, qstep_max
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    def mse_for_q(qstep):
        q = np.clip(np.round(x / qstep), -Qmax, Qmax)
        diff = x - q * qstep
        return float(np.mean(diff * diff))

    fc = mse_for_q(c)
    fd = mse_for_q(d)

    for _ in range(search_steps):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) * invphi
            fc = mse_for_q(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) * invphi
            fd = mse_for_q(d)

    qstep_opt = (a + b) / 2.0
    q = np.clip(np.round(x / qstep_opt), -Qmax, Qmax).astype(np.int32)
    return q, float(qstep_opt)

# ======================================================================
# Measure peak memory usage
# ======================================================================

def measure_peak_memory(func, *args, **kwargs):
    process = psutil.Process()
    peak = 0
    done = False

    def sampler():
        nonlocal peak
        while not done:
            mem = process.memory_info().rss
            peak = max(peak, mem)
            time.sleep(0.01)  # 10ms interval sampling

    t = threading.Thread(target=sampler)
    t.start()
    result = func(*args, **kwargs)
    done = True
    t.join()
    return result, peak

def deepcabac_encode(q_for_enc, bin_len):
    enc = deepCABAC.Encoder()
    enc.initCtxModels(bin_len, PARAM_OPT_FLAG)
    t0 = timer()
    enc.encodeLayer(
        q_for_enc,
        0, 0, 1, 0, 0,
        CHAN_SKIP_DEFAULT(q_for_enc.shape[0] if q_for_enc.ndim >= 2 else 1),
        HDSP_MODE,
        HDSP_HIST,
        0, 0
    )
    bs_arr = enc.finish()
    t1 = timer()
    encode_time = t1 - t0
    return bs_arr, encode_time

def deepcabac_decode(bs_arr, q_shape, bin_len):
    dec = deepCABAC.Decoder()
    dec.setStream(bs_arr)
    dec.initCtxModels(bin_len)
    decoded = np.zeros(q_shape, dtype=np.int32)
    t0 = timer()
    dec.decodeLayer(decoded, 0, 0, 1, 0, HDSP_MODE, HDSP_HIST, 0, 0)
    dec.finish()
    t1 = timer()
    decode_time = t1 - t0
    return decoded, decode_time

# ---------------- CSV setup ----------------
csv_path = os.path.join(OUT_DIR, "compression_results.csv")
fields = [
    "param_name", "layer_type", "shape", "numel",
    "bits_quant", "raw_bytes",
    "qstep",
    "mse_quant", "nmse_quant",
    "mse_post", "nmse_post",
    "compressed_bytes", "compression_ratio",
    "compression_gain_pct", "bits_per_element",
    "decoded_equal", "peak_enc_mem", "peak_dec_mem",
    "encode_time", "decode_time", "peak_delta_enc", "peak_delta_dec"
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

# ---------------- load model ----------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


recon_models = {8:{},12:{}}

# ---------------- main loop ----------------
for name, param in tqdm(list(model.named_parameters()), desc="params"):
    arr = param.detach().cpu().numpy().astype(np.float32)
    shape = arr.shape
    numel = int(arr.size)

    lname = name.lower()
    if "norm" in lname or "layernorm" in lname:
        layer_type = "norm"
    elif "dense" in lname:
        layer_type = "dense"
    elif "weight" in lname:
        layer_type = "weight"
    elif "bias" in lname:
        layer_type = "bias"
    else:
        layer_type = "other"

    print(f"Processing layer: {name}, type: {layer_type}, shape: {shape}")

    # variance for NMSE normalization
    var = float(np.var(arr)) if np.var(arr) > 0 else 1.0

    for bits in (8, 12):
        bin_len = 5 if bits == 8 else 8

        qIndex = np.zeros_like(arr, dtype=np.int32)
        qstep = None

        if layer_type == "weight":
            # Use DeepCABAC quantLayer for weights
            qa = QUANTLAYER_ARGS[layer_type]
            enc_q = deepCABAC.Encoder()
            try:
                enc_q.initCtxModels(bin_len, PARAM_OPT_FLAG)
                qp_return = enc_q.quantLayer(
                    arr.astype(np.float32),
                    qIndex,
                    qa["dq_flag"],
                    qa["qpDensity"],
                    qa["qp"],
                    qa["lambdaScale"],
                    bin_len,
                    qa["scan_order"],
                    qa["general_profile_idc"]
                )
                # prefer qp_return if it is meaningful (non-zero), else fallback to qa["qp"]
                qp_used = int(qp_return) if qp_return is not None and qp_return != 0 else int(qa["qp"])
                qstep = compute_qstep(qp_used, qa["qpDensity"])
            except Exception as e:
                print("QuantLayer failed for", name, bits, ":", e)
                continue
        
        else:
            # Use fast optimized uniform quant for bias/norm/other
            qIndex, qstep = optimal_uniform_quant(arr, 12, search_steps=50)

        # quantization distortion BEFORE entropy coding
        arr_quant = (qIndex.astype(np.float32)) * qstep
        mse_quant = float(np.mean((arr - arr_quant) ** 2))
        nmse_quant = mse_quant / var

        # Clip to desired bitdepth BEFORE encoding 
        clip_bit = 8 if (layer_type == "weight") else 12
        q_clipped = convert_bitdepth(qIndex, clip_bit , unsigned=False)

        # Prepare for encoding (deepCABAC expects int32)
        q_for_enc = np.ascontiguousarray(q_clipped.astype(np.int32))

        # raw_bytes = storage as int8/int12
        mult = 1 if clip_bit == 8 else 1.5
        raw_bytes = q_for_enc.size * mult

        # encode
        compressed_bytes = -1
        bs_arr = None
        peak_enc_mem = -1
        try:
            baseline_enc_mem = psutil.Process().memory_info().rss
            (bs_arr, encode_time), peak_enc_mem = measure_peak_memory(
                deepcabac_encode, q_for_enc, bin_len)
            compressed_bytes = int(bs_arr.nbytes)
        except Exception as e:
            print("Encoding failed for", name, bits, ":", e)
            compressed_bytes = -1

        # decode
        decoded_equal = False
        decoded = None
        peak_dec_mem = -1
        try:
            baseline_dec_mem = psutil.Process().memory_info().rss
            (decoded, decode_time), peak_dec_mem = measure_peak_memory(
                deepcabac_decode, bs_arr, q_for_enc.shape, bin_len)
            decoded_equal = np.array_equal(decoded, q_for_enc)
        except Exception as e:
            print("Decoding failed for", name, bits, ":", e)

        if decoded is not None:
            arr_recon_post = (decoded.astype(np.float32)) * qstep
            recon_models[bits][name] = arr_recon_post.astype(np.float32)
            mse_post = float(np.mean((arr - arr_recon_post) ** 2))
            nmse_post = mse_post / var


            orig = arr
            recon = arr_recon_post
            mse_2 = float(np.mean((orig - recon) ** 2))
            maxerr = float(np.max(np.abs(orig - recon)))
            if maxerr > 1e-3:
                print(f"⚠️ Warning: High reconstruction error in {name} (q={bits}-bit): MSE={mse_2}, MaxErr={maxerr}")
        else:
            mse_post = float("nan")
            nmse_post = float("nan")

        # compression metrics
        ratio = (raw_bytes / compressed_bytes) if (compressed_bytes and compressed_bytes > 0) else None
        gain_pct = (1 - (1 / ratio)) * 100 if ratio else None
        bpe = (compressed_bytes * 8 / numel) if compressed_bytes and compressed_bytes > 0 else None

        row = {
            "param_name": name,
            "layer_type": layer_type,
            "shape": str(shape),
            "numel": numel,
            "bits_quant": bits,
            "raw_bytes": raw_bytes,
            "qstep": float(qstep),
            "mse_quant": mse_quant,
            "nmse_quant": nmse_quant,
            "mse_post": mse_post,
            "nmse_post": nmse_post,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": ratio if ratio else "",
            "compression_gain_pct": gain_pct if gain_pct else "",
            "bits_per_element": bpe if bpe else "",
            "decoded_equal": bool(decoded_equal),
            "peak_enc_mem": peak_enc_mem,
            "peak_dec_mem": peak_dec_mem,
            "encode_time": encode_time,
            "decode_time": decode_time,
            "peak_delta_enc": peak_enc_mem - baseline_enc_mem,
            "peak_delta_dec": peak_dec_mem - baseline_dec_mem
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)

        if not decoded_equal:
            print(f"⚠️ Warning: Entropy mismatch in {name} (q={bits}-bit)")


torch.save(recon_models, os.path.join(OUT_DIR, "reconstructed_tensors.pt"))
print("💾 Saved reconstructed tensors to reconstructed_tensors.pt")

print("✅ Done! Results saved to:", csv_path)
