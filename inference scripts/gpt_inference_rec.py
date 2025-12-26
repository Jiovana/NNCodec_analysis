import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import math
import transformers
transformers.logging.set_verbosity_error()

def perplexity_sliding_window(model, tokenizer, text, max_length=512, stride=512):
    # Tokenize once (no truncation)
    enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
    input_ids = enc["input_ids"][0]  # shape: (n_tokens,)
    n_tokens = input_ids.size(0)

    model.eval()
    total_loss = 0.0
    total_count = 0

    # stride should be <= max_length; for full overlap use stride < max_length
    for begin_idx in tqdm(range(0, n_tokens, stride), desc="Chunks"):
        end_idx = min(begin_idx + max_length, n_tokens)
        input_ids_chunk = input_ids[begin_idx:end_idx]

        # prepare labels: mask context tokens with -100
        # we set labels equal to input ids, but tokens before the "target" region are -100.
        labels = input_ids_chunk.clone()
        # If begin_idx == 0, there is no "context" to mask. If we want some context-only region,
        # set prefix_length accordingly. Common approach: predict all tokens in the chunk
        # but to use more context for the rightmost tokens, use overlapping windows.
        # We'll mask the entire chunk initially and only unmask the region we want to predict:
        # Here we choose to predict the *entire* chunk; but to follow HF strict approach, you can
        # mask the first (len_chunk - stride) tokens so only newer tokens are predicted.
        # A more robust approach (from HF) is:
        chunk_len = input_ids_chunk.size(0)
        # start_predict = max(0, chunk_len - stride)  # unmask only last 'stride' tokens
        # labels[:start_predict] = -100
        # However, many implementations compute loss over the whole chunk to approximate.
        # To mimic HF recommended: unmask last (chunk_len - context_overlap) tokens.
        # For simplicity, we will unmask all tokens but normalize at the end by number of tokens used.
        # If you want HF exact approach, uncomment the three lines above and comment the 'unmask all' part.

        # Put into model (batch dimension)
        input_batch = input_ids_chunk.unsqueeze(0).to(model.device)
        labels = labels.unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_batch, labels=labels)
            # outputs.loss is the average loss over non -100 labels in this batch
            # Multiply by number of predicted tokens in this chunk to get summed NLL
            # We must count valid labels (labels != -100)
            valid = (labels != -100).sum().item()
            if valid == 0:
                continue
            nll = outputs.loss.item() * valid
            total_loss += nll
            total_count += valid

        if end_idx == n_tokens:
            break

    avg_nll = total_loss / total_count
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, total_count

if __name__ == "__main__":
    model_name = "openai-gpt"   # or your GPT-1 checkpoint
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # load reconstructed tensors
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    d = torch.load("gpt_quant_eval_mixed/reconstructed_tensors.pt", weights_only=False)
    # choose precision: 8 or 16
    bits = 8
    selected = d[bits]
    # overwrite model parameters
    with torch.no_grad():
        for name, tensor_np in selected.items():
            model.state_dict()[name].copy_(torch.from_numpy(tensor_np))
    print(f"Model GPT successfully reconstructed for {bits} bits !")
    

    # Use full validation split for robust estimate
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join([t for t in dataset["text"] if t.strip()])

    ppl, avg_nll, token_count = perplexity_sliding_window(model, tokenizer, text, max_length=512, stride=256)
    print(f"Perplexity: {ppl:.4f}, avg_nll: {avg_nll:.6f}, tokens: {token_count}")
