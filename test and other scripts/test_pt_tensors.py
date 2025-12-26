import torch, numpy as np

p = "multi_model_quant_eval/resnet50_reconstructed_tensors.pt"
d = torch.load(p, map_location="cpu")
print("Available bits:", list(d.keys()))
sel = d[8]

# print diagnostics for a few keys
for name, arr in list(sel.items())[:6]:
    # normalize to numpy for uniform inspection
    if isinstance(arr, torch.Tensor):
        a = arr.detach().cpu().numpy()
    else:
        a = np.array(arr)

    print(name, a.dtype, a.shape,
          "min", float(np.min(a)), "max", float(np.max(a)),
          "mean", float(np.mean(a)), "std", float(np.std(a)))
