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
    model_name = "textattack/bert-base-uncased-SST-2"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("sst2", split="validation[:1000]")
    acc = evaluate(model, tokenizer, dataset)
    print("Accuracy:", acc)
