import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
import torchvision.models as tvmodels

def get_model(model_name):
    if model_name == "bert":
        model_name = "textattack/bert-base-uncased-SST-2"
        return AutoModelForSequenceClassification.from_pretrained(model_name)
        #return AutoModel.from_pretrained("bert-base-uncased")
    elif model_name == "gpt":
        return AutoModelForCausalLM.from_pretrained("openai-gpt")
    elif model_name == "vit":
        return tvmodels.vit_b_16(weights=tvmodels.ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == "resnet":
        return tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == "efficientnet":
        return tvmodels.efficientnet_b0(weights=tvmodels.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def collect_histograms(model, output_dir, bins=100, max_layers=None):
    os.makedirs(output_dir, exist_ok=True)
    layer_idx = 0
    for name, param in model.named_parameters():
        #if not param.requires_grad:
         #   continue
        # Flatten the weights
        data = param.detach().cpu().numpy().flatten()
        # Optionally skip extremely large layers if you want
        if max_layers is not None and layer_idx >= max_layers:
            break

        # Compute histogram
        plt.figure(figsize=(6,4))
        plt.hist(data, bins=bins, density=True, log=True)
        plt.title(f"{name} (size={data.size})")
        plt.xlabel("Weight value")
        plt.ylabel("Density (log scale)")
        fname = os.path.join(output_dir, f"{layer_idx:03d}_{name.replace('/', '_')}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

        # Save raw data for later use
        np.save(os.path.join(output_dir, f"{layer_idx:03d}_{name.replace('/', '_')}.npy"), data)

        # Also log some info
        print(f"[{layer_idx}] {name}, parameter count = {data.size}")
        layer_idx += 1

def main():
    parser = argparse.ArgumentParser(description="Model weight histograms")
    parser.add_argument("--model", choices=["bert","gpt","vit","resnet","efficientnet"], required=True)
    parser.add_argument("--output_dir", type=str, default="histograms")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--max_layers", type=int, default=None,
                        help="max number of layers to process (for quick test)")
    args = parser.parse_args()

    model = get_model(args.model)
    model.eval()
    collect_histograms(model, os.path.join(args.output_dir, args.model), bins=args.bins,
                       max_layers=args.max_layers)

if __name__ == "__main__":
    main()
