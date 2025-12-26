#!/usr/bin/env python3
"""
resnet50_lite_nncodec_flow.py

Create a ResNet50-lite model (subset of ResNet50), run NNCodec encode/decode (full flow),
save reconstructed params and flush the deepCABAC profiler to profile.csv.

Usage:
    python resnet50_lite_nncodec_flow.py --bitdepth 8 --results ./results_lite

Requirements:
    - nncodec (python bindings providing nncodec.nn.encode / decode)
    - torchvision
    - torch
    - numpy
"""

import os
import argparse
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvmodels

# --- nncodec imports (must be in your PYTHONPATH / env) ---
from nncodec.nn import encode, decode
from nncodec.framework.pytorch_model import np_to_torch, torch_to_numpy
from nncodec.extensions import deepCABAC

# -----------------------------
# Determinism / seeds
# -----------------------------
SEED_RANDOM = 909
SEED_TORCH = 808
SEED_NUMPY = 303

os.environ["PYTHONHASHSEED"] = str(SEED_RANDOM)
random.seed(SEED_RANDOM)
np.random.seed(SEED_NUMPY)
torch.manual_seed(SEED_TORCH)
torch.cuda.manual_seed_all(SEED_TORCH)
torch.use_deterministic_algorithms(True)

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="NNCodec ResNet50-lite profiling run")
parser.add_argument("--bitdepth", type=int, default=8, choices=[8, 16], help="bitdepth to target (for some flows)")
parser.add_argument("--results", type=str, default="./results_lite", help="output directory")
parser.add_argument("--reconstructed_path", type=str, default=None, help="optional .pt to preload reconstructed tensors")
parser.add_argument("--model_backbone", type=str, default="resnet50", choices=["resnet50"], help="base model to strip")
parser.add_argument("--use_layer2",default= False, action="store_true", help="include first block of layer2 (default False reduces mem)")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

os.makedirs(args.results, exist_ok=True)

# -----------------------------
# Build ResNet50-lite
# -----------------------------
def make_resnet50_lite():
    """Return a small model that reuses portions of torchvision resnet50.

    The returned module exposes attributes: conv1, bn1, layer1, layer2 (optional),
    avgpool, fc — so it is a plausible ResNet-like model for parameter shapes.
    """
    weights = tvmodels.ResNet50_Weights.IMAGENET1K_V2
    full = tvmodels.resnet50(weights=weights)
    full.eval()

    class ResNet50Lite(nn.Module):
        def __init__(self, full_model):
            super().__init__()
            include_layer2_block=True
            # reuse initial layers
            self.conv1 = copy.deepcopy(full_model.conv1)
            self.bn1 = copy.deepcopy(full_model.bn1)
            # layer1: full
            self.layer1 = copy.deepcopy(full_model.layer1)
            # layer2: optionally only first block
            if include_layer2_block:
                # layer2 is a Sequential of Bottleneck blocks: keep first block as a Seq
                first_block = copy.deepcopy(full_model.layer2[0])
                # make layer2 a Sequential with single block (keeps module names)
                self.layer2 = nn.Sequential(first_block)
            else:
                # empty sequential for minimal model
                self.layer2 = nn.Sequential()
            # keep avgpool and fc but reduce fc input dims (safe since shapes might differ)
            self.avgpool = copy.deepcopy(full_model.avgpool)
            self.fc = copy.deepcopy(full_model.fc)
            # note: we keep everything as float32 by default

        def forward(self, x):
            # minimal forward — we don't actually run it in this profiling script,
            # but it's implemented for conformance.
            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self.layer1(x)
            if len(self.layer2) > 0:
                x = self.layer2(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    lite = ResNet50Lite(full)
    # put into eval mode and float32
    lite.eval()
    return lite

# -----------------------------
# Main flow: encode/decode using NNCodec
# -----------------------------
def main():
    print("Building ResNet50-lite (include_layer2_first_block={})".format(args.use_layer2))
    model = make_resnet50_lite()
    model.eval()

    # Optionally print param counts
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet50-lite param count: {total_params:,}")

    # Optionally preload reconstructed tensors (useful for debugging)
    if args.reconstructed_path is not None:
        try:
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        except Exception:
            # safe_globals may not be necessary; ignore if unavailable
            pass
        if os.path.exists(args.reconstructed_path):
            print("Loading reconstructed dict from:", args.reconstructed_path)
            rec_dict = torch.load(args.reconstructed_path, weights_only=False)
            if args.bitdepth in rec_dict:
                selected = rec_dict[args.bitdepth]
                with torch.no_grad():
                    sd = model.state_dict()
                    # copy matching params (names likely differ from full resnet)
                    copied = 0
                    for k, v in selected.items():
                        if k in sd:
                            sd[k].copy_(torch.from_numpy(v).to(sd[k].dtype))
                            copied += 1
                    model.load_state_dict(sd, strict=False)
                print(f"Copied {copied} reconstructed tensors into lite model.")
            else:
                print("Bitdepth key not found in reconstructed dict — skipping load.")

    # --- Run NNCodec full model encode/decode (this will traverse CABAC internals) ---
    print("Encoding model using NNCodec (full flow). This triggers CABAC internal paths.")
    args_dict = vars(args).copy()
    try:
        bitstream = encode(model, args_dict)
        print("Encode finished. Bitstream type:", type(bitstream))
    except Exception as e:
        print("ERROR during encode():", e)
        # try to still flush profiler
        try:
            print("Attempting to flush deepCABAC profiler despite encode error...")
            deepCABAC.Encoder().flushProfiler()
        except Exception:
            pass
        raise

    print("Decoding model via NNCodec.decode()...")
    try:
        rec_mdl_params = decode(bitstream, args_dict)
    except Exception as e:
        print("ERROR during decode():", e)
        # we still attempt to flush profiler
        try:
            deepCABAC.Encoder().flushProfiler()
        except Exception:
            pass
        raise

    # Convert rec_mdl_params -> torch state_dict and save
    try:
        torch_state = np_to_torch(rec_mdl_params)
        out_dict_path = os.path.join(args.results, f"resnet50_lite_dict_dec_rec.pt")
        torch.save(torch_state, out_dict_path)
        print("Saved reconstructed torch state_dict to:", out_dict_path)
    except Exception as e:
        print("ERROR saving reconstructed params:", e)

    # Save bitstream bytes to disk for inspection
    try:
        bitstream_path = os.path.join(args.results, "resnet50_lite.bitstream.npz")
        # try to save as numpy bytes or python object; encode() return may vary
        # If bitstream is a bytes-like object:
        if isinstance(bitstream, (bytes, bytearray)):
            with open(bitstream_path + ".bin", "wb") as fh:
                fh.write(bitstream)
            print("Saved bitstream to", bitstream_path + ".bin")
        else:
            # fallback: attempt numpy save for object
            np.savez_compressed(bitstream_path, bs=bitstream)
            print("Saved bitstream to", bitstream_path)
    except Exception as e:
        print("WARNING: could not save bitstream:", e)

    # -----------------------------
    # Flush profiler (C++ deepCABAC binding exposes this)
    # -----------------------------
    print("Flushing deepCABAC profiler to profile.csv (if available)...")
    try:
        # The binding provides a global flushProfiler (no encoder instance required)
        deepCABAC.Encoder().flushProfiler()
        print("Called deepCABAC.Encoder().flushProfiler()")
    except Exception as e:
        print("deepCABAC.flushProfiler() failed (binding may not expose it or OOM):", e)

    print("Done.")

if __name__ == "__main__":
    main()
