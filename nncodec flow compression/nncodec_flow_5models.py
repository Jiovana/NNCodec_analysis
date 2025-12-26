import os
import torch
import numpy as np
import random
from tqdm import tqdm
import argparse

# Torchvision + HuggingFace models
from torchvision import models
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM

# NNCodec
from nncodec.nn import encode, decode
from nncodec.framework.pytorch_model import np_to_torch
from nncodec.extensions import deepCABAC

import psutil, time, threading


# ======================================================================
# Deterministic behavior
# ======================================================================
os.environ["PYTHONHASHSEED"] = "123"
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ======================================================================
# Argument parser
# ======================================================================
parser = argparse.ArgumentParser(description="NNCodec encode/decode for 5 models")
parser.add_argument("--results", type=str, default="./nncodec_results")
parser.add_argument("--bitdepth", type=int, default=8)
parser.add_argument("--verbose", default=True, action="store_true")
args = parser.parse_args()

os.makedirs(args.results, exist_ok=True)
device = torch.device("cpu")


# ======================================================================
# Model constructors
# ======================================================================
models_to_analyze = {
    "BERT": lambda: AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    ),
    "GPT": lambda: AutoModelForCausalLM.from_pretrained(
        "openai-gpt"
    ),
    "ResNet50": lambda: models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    ),
    "EfficientNetB0": lambda: models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    ),
    "ViT": lambda: models.vit_b_16(
        weights=models.ViT_B_16_Weights.IMAGENET1K_V1
    ),
}

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

# ======================================================================
# Run encode+decode for each model
# ======================================================================
def process_model(model_name, constructor):
    print("\n" + "=" * 60)
    print(f"### Processing model: {model_name}")
    print("=" * 60)

    out_dir = os.path.join(args.results, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print(f"Loading model '{model_name}'...")
    model = constructor()
    model.to(device)
    model.eval()

    # Prepare args dictionary for NNCodec
    nnc_args = {
        "bitdepth": args.bitdepth,
        "results": out_dir,
        "verbose": True,
        # dummy fields expected by nncodec (not used in encode)
        "imagenet_val": "",
        "reconstructed_path": None,
    }

    # --------------------------------------------
    # ENCODE
    # --------------------------------------------
    print(f"\n=== Encoding {model_name} using NNCodec (DeepCABAC) ===")
    (bitstream, peak_mem_encode) = measure_peak_memory(encode, model, vars(args))

    bs_path = os.path.join(out_dir, f"{model_name}_bitstream.bin")
    with open(bs_path, "wb") as f:
        f.write(bitstream)
    print(f"Bitstream saved to {bs_path}")

    # --------------------------------------------
    # DECODE
    # --------------------------------------------
    print(f"\n=== Decoding {model_name} bitstream ===")
    (rec_params, peak_mem_decode) = measure_peak_memory(decode, bitstream, vars(args))



    
    print("Peak RAM encode : %.2f MB" % (peak_mem_encode / 1e6))
    print("Peak RAM decode : %.2f MB" % (peak_mem_decode / 1e6))
    # Save reconstructed state dict
    #torch.save(rec_params, os.path.join(out_dir, f"{model_name}_rec_params.pt"))

    #print(f"Finished model: {model_name}")
    #print("-" * 60)


# ======================================================================
# MAIN LOOP
# ======================================================================
if __name__ == "__main__":
    print("\n===============================================")
    print("Running NNCodec encode/decode for 5 models...")
    print("===============================================\n")

    for model_name, constructor in models_to_analyze.items():
        process_model(model_name, constructor)

    print("\n===============================================")
    print("All models processed successfully.")
    print("Results saved in:", args.results)
    print("===============================================")
