import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


#model.eval()
#for name, p in model.named_parameters():
#    assert p.dtype == torch.float32, f"Parameter {name} is not FP32!"
#print("✔ Model verified: all parameters are FP32")


# load reconstructed tensors
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
d = torch.load("bert_quant_eval_mixed/reconstructed_tensors.pt", weights_only=False)

# choose precision: 8 or 16
selected = d[8]

# overwrite model parameters
with torch.no_grad():
    for name, tensor_np in selected.items():
        model.state_dict()[name].copy_(torch.from_numpy(tensor_np))

print("Model successfully reconstructed!")
