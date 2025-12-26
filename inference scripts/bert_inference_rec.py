from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm

def evaluate(model, tokenizer, dataset):
    model.eval()
    correct, total = 0, 0
    for item in tqdm(dataset):
        inputs = tokenizer(item['sentence'], return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        correct += pred == item['label']
        total += 1
    return correct / total

if __name__ == "__main__":

    MODEL_NAME = "textattack/bert-base-uncased-SST-2"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # load reconstructed tensors
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    d = torch.load("bert_quant_eval_mixed/reconstructed_tensors.pt", weights_only=False)
    # choose precision: 8 or 16
    selected = d[8]
    # overwrite model parameters
    with torch.no_grad():
        for name, tensor_np in selected.items():
            model.state_dict()[name].copy_(torch.from_numpy(tensor_np))
    print("Model BERT successfully reconstructed!")


    #model_name = "textattack/bert-base-uncased-SST-2"
    #model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("sst2", split="validation[:1000]")
    acc = evaluate(model, tokenizer, dataset)
    print("Accuracy:", acc)
