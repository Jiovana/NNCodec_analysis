from nncodec.tensor import encode, decode
from nncodec.extensions import deepCABAC
import torch
import numpy as np
import os
import re
from collections import Counter



def quantize_alone():
    tensor_path = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors\K_1layer.npy"
    tensor_id = "K"
    bitdepth = 8

    tensor = torch.tensor(np.load(tensor_path))

    args = {
        "tensor_path": tensor_path,
        "tensor_id": tensor_id,
        "bitdepth": bitdepth,
        "approx_method": "uniform",  # typical
        "quantize_only": False,      # False → also entropy code
        "use_dq": False,             # optional: Trellis quantizer off
        "tca": False,                # Temporal Context Adaptation off
        "sparsity": 0.0,
        "verbose": True
    }

    print(tensor)
    bitstream = encode(tensor, args, quantize_only=False)
       # ...existing code...

     
    """ print("Inspecting bitstream items (showing up to 20):")
    # only needed if you want to check numpy types explicitly
    
    for i, (k, v) in enumerate(bitstream.items()):
        if i >= 20:
            print("...more items omitted...")
            break
        print(
            f"key[{i}]: type={type(k).__name__}, value[{i}]: type={type(v).__name__}, "
            f"is_numpy_int32={isinstance(v, getattr(np, 'int32', ())), repr(v)}"
        )
    # ...existing code...

    # save the bitstream (dict with tensor) to a file (keep .pt for compatibility)
    out_dir = os.path.dirname(tensor_path)
    base = os.path.splitext(os.path.basename(tensor_path))[0]
    out_path_pt = os.path.join(out_dir, f"{base}_{tensor_id}_quantized16.pt")
    os.makedirs(out_dir, exist_ok=True)
    try:
        torch.save(bitstream, out_path_pt)
        print(f"Saved quantized bitstream to {out_path_pt}")
    except Exception as e:
        print(f"Failed saving .pt bitstream: {e}")

    # Try to extract the quantized tensor/array from the bitstream and save as .npy
    out_path_npy = os.path.join(out_dir, f"{base}_{tensor_id}_quantized16.npy")

    def _to_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        try:
            arr = np.asarray(obj)
            if arr.size > 0:
                return arr
        except Exception:
            pass
        return None

    arr = None
    # common keys to prefer
    preferred_keys = (tensor_id, "tensor", "data", "quantized")
    if isinstance(bitstream, dict):
        # try preferred keys first
        for k in preferred_keys:
            if k in bitstream:
                arr = _to_numpy(bitstream[k])
                if arr is not None:
                    break
        # otherwise search values for a suitable array-like
        if arr is None:
            for k, v in bitstream.items():
                arr = _to_numpy(v)
                if arr is not None:
                    break

    # fallback: attempt to decode the bitstream to get the tensor
    if arr is None:
        try:
            decoded = decode(bitstream, tensor_id)
            arr = _to_numpy(decoded)
        except Exception:
            arr = None

    if arr is not None:
        try:
            np.save(out_path_npy, arr)
            print(f"Saved quantized tensor as .npy to {out_path_npy}")
        except Exception as e:
            print(f"Failed saving .npy: {e}")
    else:
        print("Could not find/extract a quantized tensor in the bitstream to save as .npy") """




    decoded = torch.tensor(decode(bitstream, args["tensor_id"]), dtype=torch.float32)


def main():
   folder = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors"
   encode_only()
   #convert_int32_folder_to_int16(folder)
   #path32 = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors\Q_1layer_Q_quantized8.npy"
   #path8 = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors\Q_1layer_Q_quantized8_int8.npy"
   #compare_int32_and_int8_files(path32, path8)
   

def test():

    import matplotlib.pyplot as plt

    folder = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors"
    tensor_ids = ("Q", "K", "V")
    bins = 100
    max_plot_points = 200000  # downsample large arrays for plotting

    # collect arrays per tensor id for an aggregated plot
    aggregates = {tid: [] for tid in tensor_ids}

    def find_tensor_id(name):
        # match standalone token like "K", or with delimiters: "K_", "_K", "-K-", etc.
        for tid in tensor_ids:
            if re.search(rf'(^|[^A-Za-z0-9]){re.escape(tid)}([^A-Za-z0-9]|$)', name):
                return tid
        return None

    def _downsample(arr):
        if arr.size > max_plot_points:
            # random sample without replacement
            idx = np.random.choice(arr.size, size=max_plot_points, replace=False)
            return arr.ravel()[idx]
        return arr.ravel()

    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        tid = find_tensor_id(base)
        if tid is None:
            continue  # skip files that are not Q/K/V

        try:
            if ext == ".npy":
                arr = np.load(path)
                src = "npy"
            else:
                continue
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        a = np.asarray(arr)
        if a.size == 0:
            print(f"{fname} ({tid}) [{src}]: empty array")
            continue

        flat = a.ravel()
        nan_mask = np.isnan(flat) if np.issubdtype(flat.dtype, np.floating) else np.zeros(flat.shape, dtype=bool)
        valid = flat[~nan_mask]
        nan_count = int(nan_mask.sum()) if nan_mask.size > 0 else 0

        unique_vals = None
        unique_count = None
        is_integer_like = np.issubdtype(flat.dtype, np.integer)
        # If float but very few unique values, treat as discrete
        try:
            unique_vals = np.unique(valid)
            unique_count = unique_vals.size
            if not is_integer_like and unique_count <= 512:
                is_integer_like = True
        except Exception:
            unique_vals = None
            unique_count = None

        mean = float(np.nanmean(flat)) if np.issubdtype(flat.dtype, np.floating) else float(flat.mean())
        std = float(np.nanstd(flat))
        med = float(np.nanmedian(flat))
        mn = float(np.nanmin(flat))
        mx = float(np.nanmax(flat))

        print(f"{fname} ({tid}) [{src}]: shape={a.shape}, dtype={a.dtype}, size={a.size}")
        print(f"  min={mn:.6g}, max={mx:.6g}, mean={mean:.6g}, std={std:.6g}, median={med:.6g}, n_nan={nan_count}")

        # append to aggregate list (store the valid flattened values)
        if valid.size > 0:
            aggregates[tid].append(valid)

        # plotting per-file
        try:
            plot_data = _downsample(valid)
            plt.figure(figsize=(8, 4))
            if is_integer_like and unique_count is not None and unique_count <= 1024:
                # discrete/bar plot
                if np.issubdtype(flat.dtype, np.integer):
                    vals, counts = np.unique(plot_data, return_counts=True)
                else:
                    vals_rounded = np.round(plot_data).astype(np.int64)
                    vals, counts = np.unique(vals_rounded, return_counts=True)
                # optionally limit number of bars for readability
                if vals.size > 500:
                    # show top 200 most frequent values
                    order = np.argsort(counts)[-200:]
                    vals = vals[order]
                    counts = counts[order]
                plt.bar(vals.astype(np.float64), counts, width=0.8, align='center')
                plt.xlabel('Value')
                plt.ylabel('Count')
                plt.title(f"{fname} ({tid}) value counts")
            else:
                # continuous histogram
                plt.hist(plot_data, bins=bins, color='C0', alpha=0.8)
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title(f"{fname} ({tid}) histogram")
            plt.tight_layout()
            plot_path = os.path.join(folder, f"{base}_{tid}_hist.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  saved histogram to {plot_path}")
        except Exception as e:
            print(f"  could not plot {fname}: {e}")

        # existing summary and info printing
        if is_integer_like and unique_count is not None and unique_count <= 1024:
            # show value counts for quantized-like tensors
            if np.issubdtype(flat.dtype, np.integer):
                vals, counts = np.unique(flat, return_counts=True)
            else:
                # round to nearest integer for display if float but discrete
                vals_rounded = np.round(flat).astype(np.int64)
                vals, counts = np.unique(vals_rounded, return_counts=True)
            # print up to first 50 unique values
            pairs = list(zip(vals.tolist(), counts.tolist()))
            pairs.sort(key=lambda x: x[0])
            display = pairs if len(pairs) <= 50 else pairs[:50]
            print(f"  unique_count={unique_count}, example value_counts (value:count): {display}")
        else:
            # continuous-like: histogram summary (already plotted)
            try:
                counts_h, bin_edges = np.histogram(flat[~np.isnan(flat)], bins=bins)
                left = list(zip(bin_edges[:-1], bin_edges[1:], counts_h))[:6]
                right = list(zip(bin_edges[:-1], bin_edges[1:], counts_h))[-6:]
                print(f"  histogram bins (first 6): {left}")
                print(f"  histogram bins (last 6): {right}")
            except Exception as e:
                print(f"  could not compute histogram: {e}")

        # optional: save a compact summary as .npz
        try:
            summary_path = os.path.join(folder, f"{base}_{tid}_summary.npz")
            np.savez_compressed(
                summary_path,
                shape=a.shape,
                dtype=str(a.dtype),
                min=mn,
                max=mx,
                mean=mean,
                std=std,
                median=med,
                unique_count=unique_count if unique_count is not None else -1,
            )
        except Exception:
            pass

    # After processing files, create aggregated plots for each tensor id
    for tid, parts in aggregates.items():
        if not parts:
            continue
        try:
            agg = np.concatenate(parts)
            if agg.size == 0:
                continue
            plot_data = _downsample(agg)
            plt.figure(figsize=(8, 4))
            # decide if aggregated data is discrete-like
            unique_vals = None
            try:
                unique_vals = np.unique(plot_data)
            except Exception:
                unique_vals = None
            if unique_vals is not None and unique_vals.size <= 1024:
                vals, counts = np.unique(plot_data, return_counts=True)
                if vals.size > 500:
                    order = np.argsort(counts)[-200:]
                    vals = vals[order]
                    counts = counts[order]
                plt.bar(vals.astype(np.float64), counts, width=0.8, align='center')
                plt.xlabel('Value')
                plt.ylabel('Count')
                plt.title(f"Aggregate {tid} value counts (all files)")
            else:
                plt.hist(plot_data, bins=bins, color='C1', alpha=0.8)
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title(f"Aggregate {tid} histogram (all files)")
            plt.tight_layout()
            agg_path = os.path.join(folder, f"{tid}_aggregate_hist.png")
            plt.savefig(agg_path)
            plt.close()
            print(f"Saved aggregate histogram for {tid} to {agg_path}")
        except Exception as e:
            print(f"Could not create aggregate plot for {tid}: {e}")

def convert_int32_npy_to_int16(npy_path, out_path=None, scale=None, clip=True, verbose=True):
    """
    Convert an int32 .npy array to int16 by simple linear scaling.
    If scale is None it will be chosen so the maximum absolute value maps to 32767.
    Returns (out_path, used_scale).
    """
    arr = np.load(npy_path)
    if arr.dtype != np.int32:
        raise ValueError(f"{npy_path}: expected dtype int32, got {arr.dtype}")

    # choose scale to map max abs to 32767 (preserve sign)
    if scale is None:
        max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        scale = 1.0 if max_abs == 0.0 else (max_abs / 32767.0)

    # avoid division by zero
    if scale == 0:
        scale = 1.0

    # scale and round, clip to int16 range, then cast
    scaled = np.round(arr / scale).astype(np.int64)  # use larger int for clipping safety
    if clip:
        scaled = np.clip(scaled, -32768, 32767)
    arr16 = scaled.astype(np.int16)

    if out_path is None:
        base, _ = os.path.splitext(npy_path)
        out_path = f"{base}_int16.npy"

    np.save(out_path, arr16)
    if verbose:
        print(f"Converted {npy_path} (int32) -> {out_path} (int16), scale={scale:.6g}")
    return out_path, scale


def convert_int32_folder_to_int16(folder, suffix="_int16", recursive=False, verbose=True):
    """
    Convert all .npy files in folder that are dtype int32 to int16.
    Only processes files whose filename (without extension) contains the substring '16'.
    Files are saved next to originals with suffix appended before extension.
    Returns list of (src, dst, scale).
    """
    results = []
    for root, dirs, files in os.walk(folder):
        for fname in sorted(files):
            if not fname.lower().endswith(".npy"):
                continue
            base_name = os.path.splitext(fname)[0]
            # only process files that contain the substring '16' in their name
            if '16' not in base_name:
                continue
            src = os.path.join(root, fname)
            try:
                arr = np.load(src, mmap_mode='r')
                if arr.dtype != np.int32:
                    continue
            except Exception:
                continue
            base, _ = os.path.splitext(src)
            dst = f"{base}{suffix}.npy"
            try:
                out, scale = convert_int32_npy_to_int16(src, out_path=dst, verbose=verbose)
                results.append((src, out, scale))
            except Exception as e:
                if verbose:
                    print(f"Failed converting {src}: {e}")
        if not recursive:
            break
    return results

def compare_int32_and_int8_files(path_int32, path_int8, scale=None, clip=True, verbose=True, report_limit=10):
    """
    Compare two .npy files: one int32 and one int8 that should represent the same values
    after linear quantization. If scale is not given it will be inferred from data
    using a few heuristics (default = max_abs_int32 / 127). The function returns
    (matches, info) where matches is True if all elements match after applying the
    chosen scale and quantization rounding+clipping, and info is a dict with details.
    """
    a32 = np.load(path_int32)
    a8 = np.load(path_int8)

    info = {"path_int32": path_int32, "path_int8": path_int8, "dtype32": str(a32.dtype), "dtype8": str(a8.dtype)}

    # Normalize shapes: require same number of elements; allow different shapes if sizes equal
    if a32.size != a8.size:
        info.update({"match": False, "reason": "size_mismatch", "size32": int(a32.size), "size8": int(a8.size)})
        if verbose:
            print(f"Size mismatch: int32 has {a32.size} elements, int8 has {a8.size}")
        return False, info

    # Ensure dtypes are expected (but continue if not)
    if a32.dtype != np.int32 and verbose:
        print(f"Warning: expected int32 for {path_int32}, got {a32.dtype}")
    if a8.dtype != np.int8 and verbose:
        print(f"Warning: expected int8 for {path_int8}, got {a8.dtype}")

    flat32 = np.asarray(a32).ravel().astype(np.int64)  # use larger int for safe division
    flat8 = np.asarray(a8).ravel().astype(np.int64)

    max_abs32 = float(np.max(np.abs(flat32))) if flat32.size > 0 else 0.0
    max_abs8 = int(np.max(np.abs(flat8))) if flat8.size > 0 else 0

    # Build candidate scales (if user didn't supply one)
    candidates = []
    if scale is not None:
        candidates = [float(scale)]
    else:
        if max_abs32 == 0:
            candidates = [1.0]
        else:
            candidates.append(max_abs32 / 127.0)            # default used by conversion helper
            if max_abs8 > 0:
                candidates.append(max_abs32 / float(max_abs8))  # ratio of maxes
            # try a robust per-element median ratio where int8 != 0
            nonzero_mask = flat8 != 0
            if np.any(nonzero_mask):
                ratios = flat32[nonzero_mask].astype(np.float64) / flat8[nonzero_mask].astype(np.float64)
                med = float(np.median(ratios))
                if np.isfinite(med) and med > 0:
                    candidates.append(med)
            # de-duplicate near-equal candidates
        # unique by rounding
        seen = []
        uniq = []
        for c in candidates:
            key = round(float(c), 12)
            if key not in seen:
                seen.append(key)
                uniq.append(float(c))
        candidates = uniq

    if verbose:
        print(f"Trying {len(candidates)} candidate scale(s): {candidates}")

    best = {"scale": None, "mismatches": None, "mismatch_idx": None, "predicted": None}
    for sc in candidates:
        sc = float(sc) if sc != 0 else 1.0
        # predicted quantized int8 from int32 using same logic as conversion
        pred = np.round(flat32 / sc).astype(np.int64)
        if clip:
            pred = np.clip(pred, -128, 127)
        pred8 = pred.astype(np.int8)
        neq = (pred8.astype(np.int64) != flat8)
        n_mismatch = int(np.count_nonzero(neq))
        if best["mismatches"] is None or n_mismatch < best["mismatches"]:
            best.update({
                "scale": sc,
                "mismatches": n_mismatch,
                "predicted": pred8.copy(),
                "mismatch_idx": np.nonzero(neq)[0]
            })
        if verbose:
            pct = 100.0 * n_mismatch / flat8.size if flat8.size > 0 else 0.0
            print(f" scale={sc:.6g}: mismatches={n_mismatch} ({pct:.4g}%)")
            if n_mismatch == 0:
                print("  -> perfect match with this scale")
                break

    info.update({
        "tried_scales": candidates,
        "chosen_scale": best["scale"],
        "mismatches": int(best["mismatches"]) if best["mismatches"] is not None else None,
        "total": int(flat8.size)
    })

    matches = (best["mismatches"] == 0)
    info["match"] = matches

    if not matches and verbose:
        idxs = best["mismatch_idx"]
        nshow = min(report_limit, idxs.size if idxs is not None else 0)
        print(f"Found {best['mismatches']} mismatches (chosen scale {best['scale']}). Showing up to {nshow}:")
        for i in range(nshow):
            idx = int(idxs[i])
            sidx = np.unravel_index(idx, a32.shape)
            v32 = int(flat32[idx])
            v8 = int(flat8[idx])
            vp = int(best["predicted"][idx]) if best["predicted"] is not None else None
            print(f" idx(flat)={idx}, idx(shape)={sidx}: int32={v32}, int8={v8}, predicted={vp}")
        if idxs.size > nshow:
            print(f" ... and {idxs.size - nshow} more mismatches")

    return matches, info


def encode_only():

    q_params = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors\Q_1layer_Q_quantized16.npy"
    q_tensor = torch.tensor(np.load(q_params))
    k_params = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors\K_1layer_K_quantized16.npy"
    k_tensor = torch.tensor(np.load(k_params))
    v_params = r"C:\Users\gomes\OneDrive\Documentos\nncodec2\nncodec2\bert_tensors\V_1layer_V_quantized16.npy"
    v_tensor = torch.tensor(np.load(v_params))

    encoder = deepCABAC.Encoder()
    encoder.initCtxModels(11,True)

    chan_skip_list = np.zeros(1, dtype=np.int32)  # or length = tensor.shape[0]
    hdsp_mode = deepCABAC.HdspMode.AlwaysOff
    hdsp_hist = np.zeros(256, dtype=np.int8)

    encoder.encodeLayer(
         v_tensor, 
         0, #dq_flag 
         0, #scan order
         1, #general_profile_idc,
         0, #parent_node_id_present_flag,
         0, #rowSkipFlag, 
         chan_skip_list, #array([0])
         hdsp_mode,
         hdsp_hist, #[<HdspMode.AlwaysOff: 99>, array([0, 0], dtype=int8)]
         0,  #codebook_size,
         0 )  #codebook_zero_offset
    
    bitstream = encoder.finish()
    # Display compression info
    print(f"Original size: {q_tensor.numel() * 4} bytes")
    print(f"Compressed size: {len(bitstream)} bytes")
    print(f"Compression ratio: {(q_tensor.numel() * 4) / len(bitstream):.2f}x")
        





if __name__ == '__main__':
    main()