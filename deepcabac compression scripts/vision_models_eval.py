# multi_model_quant_eval_uniform_with_buffers.py
import os
import csv
from tqdm import tqdm
import numpy as np
import torch
import torchvision.models as tvmodels
from nncodec.extensions import deepCABAC
import psutil, time, threading
from timeit import default_timer as timer

# Output
OUT_DIR = "multi_model_quant_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# Models to test: keys -> loader lambdas
MODEL_LOADERS = {
    "resnet50": lambda: tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V1),
    "efficientnet_b0": lambda: tvmodels.efficientnet_b0(weights=tvmodels.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "vit_b16": lambda: tvmodels.vit_b_16(weights=tvmodels.ViT_B_16_Weights.IMAGENET1K_V1),
}

# DeepCABAC config
CHAN_SKIP_DEFAULT = lambda num_ch: np.zeros(num_ch if num_ch > 0 else 1, dtype=np.int32)
HDSP_MODE = deepCABAC.HdspMode.AlwaysOff
HDSP_HIST = np.zeros(256, dtype=np.int8)
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

# ---------------- optimized uniform quantizer ----------------
def optimal_uniform_quant(x, bitwidth, search_steps=40):
    """
    Per-tensor symmetric uniform quantization optimized (fast).
    Returns (q_int32, qstep).
    """
    x = x.astype(np.float32)
    Qmax = (1 << (bitwidth - 1)) - 1

    if x.size == 0 or np.all(x == 0):
        return np.zeros_like(x, dtype=np.int32), 1.0

    std = float(np.std(x))
    if std == 0:
        return np.zeros_like(x, dtype=np.int32), 1.0

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

def convert_bitdepth(q_int32, bitwidth, unsigned=False):
    """Clip quantized integers to target bitwidth (signed/unsigned)."""
    if unsigned:
        qmin = 0
        qmax = (1 << bitwidth) - 1
    else:
        qmin = -(1 << (bitwidth - 1))
        qmax = (1 << (bitwidth - 1)) - 1
    return np.clip(q_int32, qmin, qmax).astype(np.int32)

def compute_qstep(qp, qpDensity):
    """Replicate the QP -> qStep mapping used in quantLayer."""
    k = 1 << qpDensity
    mul = k + (qp & (k - 1))
    shift = int(qp) >> qpDensity
    qstep = float(mul) * (2.0 ** (shift - qpDensity))
    return qstep

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
            time.sleep(0.001)  # 1ms interval sampling

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

def bypass_encode(q_for_enc, bin_len, unsigned_possible):
        enc_bp = deepCABAC.Encoder()
        enc_bp.initCtxModels(bin_len, PARAM_OPT_FLAG)

        # unsigned or signed?
        t0 = timer()
        if unsigned_possible:
            for v in q_for_enc.reshape(-1):
                enc_bp.uae_v(bits, int(v))
        else:
            for v in q_for_enc.reshape(-1):
                enc_bp.iae_v(bits, int(v))
        t1 = timer()

        bp_bs = enc_bp.finish()
        encode_time = t1 - t0
        return bp_bs, encode_time

def bypass_decode(bs_arr, q_shape, bin_len, unsigned_possible):
     # manual EP decoding (bin-by-bin)
    dec = deepCABAC.Decoder()
    dec.setStream(bs_arr)
    dec.initCtxModels(bin_len)
    decoded = np.zeros(q_shape, dtype=np.int32)

    it = decoded.reshape(-1)
    t0 = timer()
    if unsigned_possible:
        for i in range(it.shape[0]):
            # unsigned bypass decode
            it[i] = dec.uae_v(bits)
    else:
        for i in range(it.shape[0]):
            it[i] = dec.iae_v(bits)
    t1 = timer()
    dec.finish()
    decode_time = t1 - t0
    return decoded, decode_time

# ---------------- main loop per model ----------------
for model_key, loader in MODEL_LOADERS.items():
    print(f"\n=== Processing model: {model_key} ===")
    try:
        model = loader()
    except Exception as e:
        print(f"Failed to load model {model_key}: {e}")
        continue

    model.eval()

    csv_path = os.path.join(OUT_DIR, f"{model_key}_compression.csv")
    fields = [
        "param_name", "layer_type", "shape", "numel",
        "bits_quant", "raw_bytes",
        "qstep",
        "mse_quant", "nmse_quant",
        "mse_post", "nmse_post",
        "compressed_bytes", "compression_ratio",
        "compression_gain_pct", "bits_per_element",
        "decoded_equal", "mode_used", "peak_enc_mem", "peak_dec_mem",
        "encode_time", "decode_time", "peak_delta_enc", "peak_delta_dec"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    reconstructed = {8: {}, 16: {}}

    encode_time = None
    decode_time = None
    encode_time_bp = None
    decode_time_bp = None
    peak_enc_mem = None
    peak_dec_mem = None
    peak_enc_mem_bp = None
    peak_dec_mem_bp = None

    # 1) Process named_parameters (trainable params)
    for name, param in tqdm(list(model.named_parameters()), desc=f"params-{model_key}"):
        arr = param.detach().cpu().numpy().astype(np.float32)
        shape = arr.shape
        numel = int(arr.size)

        lname = name.lower()
        if "bias" in lname:
            layer_type = "bias"
        elif "weight" in lname:
            layer_type = "weight"
        else:
            layer_type = "other"

        var = float(np.var(arr)) if np.var(arr) > 0 else 1.0

        for bits in (8, 16):
            bin_len = 5 if bits == 8 else 8

            qIndex = np.zeros_like(arr, dtype=np.int32)
            qstep = 1.0

            if layer_type == "weight" and numel >= 10000:
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
                    qp_used = int(qp_return) if qp_return is not None and qp_return != 0 else int(qa["qp"])
                    qstep = compute_qstep(qp_used, qa["qpDensity"])
                except Exception as e:
                    print("QuantLayer failed for", name, bits, ":", e)
                    # fallback to uniform
                    qIndex, qstep = optimal_uniform_quant(arr, bits, search_steps=50)
            else:
                # Use fast optimized uniform quant for bias/norm/other
                qIndex, qstep = optimal_uniform_quant(arr, bits, search_steps=50)

            # compute distortion BEFORE encoding (reconstruct using qstep)
            arr_quant = (qIndex.astype(np.float32)) * qstep
            mse_quant = float(np.mean((arr - arr_quant) ** 2))
            nmse_quant = mse_quant / var

            # Clip to desired bitdepth and choose unsigned if possible
            unsigned_possible = np.all(qIndex >= 0)
            q_clipped = convert_bitdepth(qIndex, bits, unsigned=unsigned_possible)

            # prepare for encoding (deepCABAC expects int32)
            q_for_enc = np.ascontiguousarray(q_clipped.astype(np.int32))

            ###########################################
            # TWO-PASS: REGULAR CABAC + MANUAL BYPASS #
            ###########################################

            raw_bytes = q_for_enc.size * (1 if bits == 8 else 2)

            # ---------- PASS 1: Regular CABAC ----------
            reg_bytes = None
            reg_bs = None
            peak_enc_mem = -1
            try:
                baseline_enc_mem = psutil.Process().memory_info().rss
                (reg_bs, encode_time), peak_enc_mem = measure_peak_memory(
                    deepcabac_encode, q_for_enc, bin_len)
                reg_bytes = int(reg_bs.nbytes)
            except Exception as e:
                print(f"[CABAC] Encode failed for {name} ({bits}-bit): {e}")
                reg_bytes = None


            # ---------- PASS 2: Manual BYPASS mode ----------
            bp_bytes = None
            bp_bs = None
            peak_enc_mem_bp = -1
            try:
                baseline_enc_mem_bp = psutil.Process().memory_info().rss
                (bp_bs, encode_time_bp), peak_enc_mem_bp = measure_peak_memory(
                    bypass_encode, q_for_enc, bin_len, unsigned_possible)
                bp_bytes = int(bp_bs.nbytes)
            except Exception as e:
                print(f"[BYPASS] Encode failed for {name} ({bits}-bit): {e}")
                bp_bytes = None


            # ---------- SELECT BETTER RESULT ----------
            use_bypass = False

            if reg_bytes is None and bp_bytes is not None:
                use_bypass = True
            elif reg_bytes is not None and bp_bytes is None:
                use_bypass = False
            elif reg_bytes is not None and bp_bytes is not None:
                # choose best compression (lowest bytes)
                use_bypass = bp_bytes < reg_bytes
            else:
                print(f"❌ Both CABAC and BYPASS failed for {name}")
                compressed_bytes = -1
                bs_arr = None
                mode_used = "none"


            if use_bypass:
                compressed_bytes = bp_bytes
                bs_arr = bp_bs
                mode_used = "bypass"
            else:
                compressed_bytes = reg_bytes
                bs_arr = reg_bs
                mode_used = "regular"

            final_encode_time = encode_time_bp if use_bypass else encode_time
            final_peak_enc_mem = peak_enc_mem_bp if use_bypass else peak_enc_mem
            final_baseline_enc_mem = baseline_enc_mem_bp if use_bypass else baseline_enc_mem

            print(f"Model: {model_key}, Param: {name}, Bits: {bits}, Mode: {mode_used}, Raw bytes: {raw_bytes}, Compressed bytes: {compressed_bytes}")
            print(f"Baseline enc mem: {final_baseline_enc_mem}, Peak enc mem: {final_peak_enc_mem}")
            ####################################
            # DECODING USING THE CHOSEN METHOD #
            ####################################
            decoded_equal = False
            decoded = None
            peak_dec_mem = -1
            try:
                if mode_used == "regular":
                   baseline_dec_mem = psutil.Process().memory_info().rss
                   (decoded, decode_time), peak_dec_mem = measure_peak_memory( 
                    deepcabac_decode, bs_arr, q_for_enc.shape, bin_len)

                elif mode_used == "bypass":
                    baseline_dec_mem = psutil.Process().memory_info().rss
                    (decoded, decode_time), peak_dec_mem = measure_peak_memory(
                        bypass_decode, bs_arr, q_for_enc.shape, bin_len, unsigned_possible)
                   

                decoded_equal = np.array_equal(decoded, q_for_enc)

            except Exception as e:
                print(f"Decoding failed for {name} ({bits}-bit): {e}")
                decoded = None

            if decoded is not None:
                arr_recon_post = (decoded.astype(np.float32)) * qstep
                reconstructed[bits][name] = arr_recon_post.astype(np.float32)
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

            ratio = (raw_bytes / compressed_bytes) if (compressed_bytes and compressed_bytes > 0) else None
            gain_pct = (1 - (1 / ratio)) * 100 if ratio else None
            bpe = (compressed_bytes * 8 / numel) if compressed_bytes and compressed_bytes > 0 else None

            final_decode_time = decode_time
            final_peak_dec_mem = peak_dec_mem    

            print(f"Model: {model_key}, Param: {name}, Bits: {bits}, Decoded equal: {decoded_equal}, Compression ratio: {ratio}, Gain %: {gain_pct}, BPE: {bpe}")

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
                "mode_used": mode_used,
                "peak_enc_mem": final_peak_enc_mem,
                "peak_dec_mem": final_peak_dec_mem,
                "encode_time": final_encode_time,
                "decode_time": final_decode_time,
                "peak_delta_enc": final_peak_enc_mem - final_baseline_enc_mem,
                "peak_delta_dec": final_peak_dec_mem - baseline_dec_mem,
            }

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(row)

            if not decoded_equal:
                print(f"⚠️ Warning: Entropy mismatch in {name} (q={bits}-bit)")

    # ------------------- 2) Process named_buffers -------------------
    buffer_csv_path = os.path.join(OUT_DIR, f"{model_key}_buffers_compression.csv")
    with open(buffer_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    for name, buf in tqdm(list(model.named_buffers()), desc=f"buffers-{model_key}"):

        arr = buf.detach().cpu().numpy()

        # skip empty buffers
        if arr.size == 0:
            continue

        # integer buffers → do not quantize (store as float for reconstruction)
        if np.issubdtype(arr.dtype, np.integer):
            for bits in (8, 16):
                reconstructed[bits][name] = arr.astype(np.float32)
            continue

        # float buffer → quantize normally
        arr = arr.astype(np.float32)
        shape = arr.shape
        numel = int(arr.size)
        layer_type = "buffer"

        var = float(np.var(arr)) if np.var(arr) > 0 else 1.0

        for bits in (8, 16):

            # quantization
            qIndex, qstep = optimal_uniform_quant(arr, bits, search_steps=50)

            arr_quant = (qIndex.astype(np.float32)) * qstep
            mse_quant = float(np.mean((arr - arr_quant) ** 2))
            nmse_quant = mse_quant / var

            # choose unsigned if possible
            unsigned_possible = np.all(qIndex >= 0)
            q_clipped = convert_bitdepth(qIndex, bits, unsigned=unsigned_possible)

            q_for_enc = np.ascontiguousarray(q_clipped.astype(np.int32))
            raw_bytes = q_for_enc.size * (1 if bits == 8 else 2)

            ###########################################
            # TWO-PASS: REGULAR CABAC + MANUAL BYPASS #
            ###########################################

            raw_bytes = q_for_enc.size * (1 if bits == 8 else 2)

            # ---------- PASS 1: Regular CABAC ----------
            reg_bytes = None
            reg_bs = None
            try:
                enc_reg = deepCABAC.Encoder()
                enc_reg.initCtxModels(bin_len, PARAM_OPT_FLAG)
                enc_reg.encodeLayer(
                    q_for_enc,
                    0, 0, 1, 0, 0,
                    CHAN_SKIP_DEFAULT(q_for_enc.shape[0] if q_for_enc.ndim >= 2 else 1),
                    HDSP_MODE,
                    HDSP_HIST,
                    0, 0
                )
                reg_bs = enc_reg.finish()
                reg_bytes = int(reg_bs.nbytes)
            except Exception as e:
                print(f"[CABAC] Encode failed for {name} ({bits}-bit): {e}")
                reg_bytes = None


            # ---------- PASS 2: Manual BYPASS mode ----------
            bp_bytes = None
            bp_bs = None
            try:
                enc_bp = deepCABAC.Encoder()
                enc_bp.initCtxModels(bin_len, PARAM_OPT_FLAG)

                # unsigned or signed?
                if unsigned_possible:
                    for v in q_for_enc.reshape(-1):
                        enc_bp.uae_v(bits, int(v))
                else:
                    for v in q_for_enc.reshape(-1):
                        enc_bp.iae_v(bits, int(v))

                bp_bs = enc_bp.finish()
                bp_bytes = int(bp_bs.nbytes)
            except Exception as e:
                print(f"[BYPASS] Encode failed for {name} ({bits}-bit): {e}")
                bp_bytes = None


            # ---------- SELECT BETTER RESULT ----------
            use_bypass = False

            if reg_bytes is None and bp_bytes is not None:
                use_bypass = True
            elif reg_bytes is not None and bp_bytes is None:
                use_bypass = False
            elif reg_bytes is not None and bp_bytes is not None:
                # choose best compression (lowest bytes)
                use_bypass = bp_bytes < reg_bytes
            else:
                print(f"❌ Both CABAC and BYPASS failed for {name}")
                compressed_bytes = -1
                bs_arr = None
                mode_used = "none"


            if use_bypass:
                compressed_bytes = bp_bytes
                bs_arr = bp_bs
                mode_used = "bypass"
            else:
                compressed_bytes = reg_bytes
                bs_arr = reg_bs
                mode_used = "regular"


            ####################################
            # DECODING USING THE CHOSEN METHOD #
            ####################################
            decoded_equal = False
            decoded = None

            try:
                if mode_used == "regular":
                    dec = deepCABAC.Decoder()
                    dec.setStream(bs_arr)
                    dec.initCtxModels(bin_len)
                    decoded = np.zeros_like(q_for_enc, dtype=np.int32)
                    dec.decodeLayer(
                        decoded, 0, 0, 1, 0, HDSP_MODE, HDSP_HIST, 0, 0
                    )
                    dec.finish()

                elif mode_used == "bypass":
                    # manual EP decoding (bin-by-bin)
                    dec = deepCABAC.Decoder()
                    dec.setStream(bs_arr)
                    dec.initCtxModels(bin_len)
                    decoded = np.zeros_like(q_for_enc, dtype=np.int32)

                    it = decoded.reshape(-1)
                    if unsigned_possible:
                        for i in range(it.shape[0]):
                            # unsigned bypass decode
                            it[i] = dec.uae_v(bits)
                    else:
                        for i in range(it.shape[0]):
                            it[i] = dec.iae_v(bits)

                    dec.finish()

                decoded_equal = np.array_equal(decoded, q_for_enc)

            except Exception as e:
                print(f"Decoding failed for {name} ({bits}-bit): {e}")
                decoded = None

            # reconstruction
            if decoded is not None:
                arr_recon_post = (decoded.astype(np.float32)) * qstep
                reconstructed[bits][name] = arr_recon_post.astype(np.float32)
                mse_post = float(np.mean((arr - arr_recon_post) ** 2))
                nmse_post = mse_post / var
            else:
                reconstructed[bits][name] = arr.astype(np.float32)
                mse_post = float("nan")
                nmse_post = float("nan")

            # compression metrics
            ratio = (raw_bytes / compressed_bytes) if (compressed_bytes and compressed_bytes > 0) else None
            gain_pct = (1 - (1 / ratio)) * 100 if ratio else None
            bpe = (compressed_bytes * 8 / numel) if compressed_bytes and compressed_bytes > 0 else None

            # write CSV row
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
                "mode_used": mode_used,
            }

            with open(buffer_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(row)


    # --- Save reconstructed arrays (use .npz to avoid pickling issues) ---
    csv_name_npz = f"{model_key}_reconstructed_tensors.npz"
    save_dict = {}
    for bits, rec in reconstructed.items():
        # make a flat dict per bits with keys names encoded safely (replace / with __)
        for k, v in rec.items():
            key = f"b{bits}__{k}"
            # ensure contiguous float32 numpy
            save_dict[key] = np.ascontiguousarray(v.astype(np.float32))

    np.savez_compressed(os.path.join(OUT_DIR, csv_name_npz), **save_dict)
    print(f"💾 Saved reconstructed tensors to {csv_name_npz}")

    # Optionally also save a torch-serializable version (commented out - may require safe globals to load)
    # torch.save({bits: {k: torch.tensor(v, dtype=torch.float32) for k, v in rec.items()} for bits, rec in reconstructed.items()},
    #            os.path.join(OUT_DIR, f"{model_key}_reconstructed_tensors.pt"))

    print(f"Done for model: {model_key}. CSV saved to: {csv_path}")
