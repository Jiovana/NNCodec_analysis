import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.serialization

def recommend_batch_size(model_name, device="cpu"):
    if device == "cpu":
        if "resnet" in model_name:
            return 128      # good balance
        if "efficientnet" in model_name:
            return 256
        if "vit" in model_name:
            return 128
    return 64  # safe fallback


def main():
    torch.set_num_threads(8)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")
    imagenet_val_dir = r"C:\Users\gomes\OneDrive\Documentos\imagenet\ILSVRC\Data\CLS-LOC\val"
    
    
   # === Select model ===
    model_name = "vit_b_16"  # or efficientnet_b0, vit_b_16

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)

    else:
        raise ValueError("Unknown model name")

    model.eval()

    # === Use native transforms ===
    transform = weights.transforms()

    # === Dataset ===
    dataset = datasets.ImageFolder(
        root=imagenet_val_dir,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=recommend_batch_size(model_name, device), shuffle=False, num_workers=8)


    # load reconstructed tensors
    #torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    npz = np.load("multi_model_quant_eval/resnet50_reconstructed_tensors.npz")
    # choose precision: 8 or 16
    bits = 8
    prefix = f"{bits}bits_"

    # build dict matching state_dict keys
    loaded = {}
    for k in npz.files:
        if not k.startswith(prefix):
            continue
        name = k[len(prefix):]            # get original key
        arr = npz[k].astype(np.float32)
        loaded[name] = arr

    sd = model.state_dict()

    # overwrite matching keys
    for name, val in loaded.items():
        if name in sd:
            sd[name] = torch.from_numpy(val).to(dtype=sd[name].dtype)
        else:
            print("WARNING: stored key not in model:", name)

    model.load_state_dict(sd, strict=True)
    model.eval()
   
    print(f"Model {model_name} successfully reconstructed for {bits} bits !")

 

    # === Inference ===
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Inference {model_name}"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Top-1
            _, pred1 = outputs.topk(1, dim=1)
            top1_correct += (pred1.squeeze() == labels).sum().item()

            # Top-5
            _, pred5 = outputs.topk(5, dim=1)
            top5_correct += sum([labels[i] in pred5[i] for i in range(len(labels))])

            total += labels.size(0)

    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100

    print(f"{model_name} Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"{model_name} Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    main()
