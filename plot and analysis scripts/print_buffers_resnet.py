import torch
import torchvision.models as models

def print_tensor_stats(t, indent="    "):
    print(f"{indent}dtype:    {t.dtype}")
    print(f"{indent}shape:    {tuple(t.shape)}")
    print(f"{indent}numel:    {t.numel()}")
    if t.dtype in (torch.float16, torch.float32, torch.float64):
        print(f"{indent}min/max:  {float(t.min()):.6f} / {float(t.max()):.6f}")
        print(f"{indent}mean/std: {float(t.mean()):.6f} / {float(t.std()):.6f}")
    itemsize = t.element_size()
    print(f"{indent}size:     {t.numel() * itemsize} bytes")
    print()

def main():
    print("\n=== Loading ResNet-50 ===")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    print("\n=== BUFFER LIST ===")
    for name, buf in model.named_buffers():
        print(f"\nBuffer: {name}")
        print_tensor_stats(buf)

    print("\n=== TOTAL BUFFERS ===")
    total = sum(buf.numel() * buf.element_size() for _, buf in model.named_buffers())
    print(f"Total buffer memory: {total / 1024:.2f} KB")

if __name__ == "__main__":
    main()
