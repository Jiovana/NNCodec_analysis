import os
import torch
import random
import numpy as np
import copy
import argparse
import warnings
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from nncodec.nn import encode, decode
from nncodec.framework.pytorch_model import np_to_torch, torch_to_numpy

# -----------------------------
# Seeds for determinism
# -----------------------------
SEED_RANDOM = 909
SEED_TORCH = 808
SEED_NUMPY = 303

os.environ["PYTHONHASHSEED"] = str(SEED_RANDOM)
torch.manual_seed(SEED_TORCH)
torch.cuda.manual_seed_all(SEED_TORCH)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
random.seed(SEED_RANDOM)
np.random.seed(SEED_NUMPY)

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description='NNCodec BERT SST-2')
parser.add_argument('--model_name', type=str, default='textattack/bert-base-uncased-SST-2')
parser.add_argument('--dataset_split', type=str, default='validation[:1000]')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--results', type=str, default='./results')
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--uc', type=int, default=0)
#parser.add_argument('--qp', type=int, default=-32)
parser.add_argument('--bitdepth', type=int, default=8)
parser.add_argument('--reconstructed_path', type=str, default=None)
parser.add_argument('--verbose', action="store_true")
args = parser.parse_args()

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and args.cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

if not os.path.exists(args.results):
    os.makedirs(args.results)

# -----------------------------
# Load BERT model + tokenizer
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model.to(device)
model.eval()

# -----------------------------
# Optionally load reconstructed tensors
# -----------------------------
if args.reconstructed_path is not None and os.path.exists(args.reconstructed_path):
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    reconstructed_dict = torch.load(args.reconstructed_path, weights_only=False)
    # Choose precision (8-bit or 16-bit)
    selected = reconstructed_dict[8]
    # Overwrite model parameters
    with torch.no_grad():
        for name, tensor_np in selected.items():
            model.state_dict()[name].copy_(torch.from_numpy(tensor_np))
    print("Model BERT successfully reconstructed!")

# -----------------------------
# Load SST-2 dataset
# -----------------------------
dataset = load_dataset("sst2", split=args.dataset_split)

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(model, tokenizer, dataset):
    model.eval()
    correct, total = 0, 0
    for item in tqdm(dataset, disable=not args.verbose):
        inputs = tokenizer(item['sentence'], return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        correct += pred == item['label']
        total += 1
    return correct / total

# -----------------------------
# NNCodec full model encode/decode
# -----------------------------
if args.uc == 0:
    # Encode and decode entire model
    bitstream = encode(model, vars(args))
    rec_mdl_params = decode(bitstream, vars(args))

    # Load reconstructed parameters into model
    model.load_state_dict(np_to_torch(rec_mdl_params))
    torch.save(model.state_dict(), f"{args.results}/{args.model_name.replace('/', '_')}_dict_dec_rec.pt")

# -----------------------------
# Evaluate reconstructed model
# -----------------------------
acc = evaluate(model, tokenizer, dataset)
print(f"Reconstructed BERT SST-2 test accuracy: {acc*100:.2f}%")
