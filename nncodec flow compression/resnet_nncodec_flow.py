import os
import torch
import numpy as np
import random
from tqdm import tqdm
import argparse

from torchvision import models, transforms, datasets

# NNCodec
from nncodec.nn import encode, decode
from nncodec.framework.pytorch_model import np_to_torch
from nncodec.extensions import deepCABAC


# -------------------------------------------------------
# Seeds for deterministic behavior
# -------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "123"
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------------
# CLI args
# -------------------------------------------------------
imagenet_val_dir = r"C:\Users\gomes\OneDrive\Documentos\imagenet\ILSVRC\Data\CLS-LOC\val"

parser = argparse.ArgumentParser(description="NNCodec ResNet50 ImageNet")
parser.add_argument("--imagenet_val", type=str, default=imagenet_val_dir,
                    help="Path to ImageNet validation directory")
parser.add_argument("--results", type=str, default="./results_resnet50")
parser.add_argument("--bitdepth", type=int, default=8)
parser.add_argument("--reconstructed_path", type=str, default=None)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

os.makedirs(args.results, exist_ok=True)

device = torch.device("cpu")


# -------------------------------------------------------
# Load ResNet50 model
# -------------------------------------------------------
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)

model.to(device)
model.eval()

print("Loaded EfficientNet-B0 with ImageNet1k pretrained weights.")


# -------------------------------------------------------
# Optional — load reconstructed model
# -------------------------------------------------------
if args.reconstructed_path is not None and os.path.exists(args.reconstructed_path):
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    loaded = torch.load(args.reconstructed_path, weights_only=False)

    # choose bitdepth
    selected = loaded[args.bitdepth]

    with torch.no_grad():
        for name, arr in selected.items():
            model.state_dict()[name].copy_(torch.from_numpy(arr))

    print(f"Restored EfficientNet-B0 parameters ({args.bitdepth}-bit quantized).")


# -------------------------------------------------------
# FULL NNCodec ENCODE + DECODE
# -------------------------------------------------------
print("\n=== Encoding model using NNCodec (DeepCABAC) ===")
bitstream = encode(model, vars(args))

print("Encoding complete. Now decoding...")

rec_params = decode(bitstream, vars(args))

print("Decoding complete.")


