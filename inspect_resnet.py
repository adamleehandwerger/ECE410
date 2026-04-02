import torch
import torchvision.models as models

# Load ResNet-18
model = models.resnet18(pretrained=False)

# Print model architecture
print(model)

# Get model summary
print("\n" + "="*50)
print("Model Summary")
print("="*50)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Inspect specific layers
print("\n" + "="*50)
print("Layer Details")
print("="*50)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        print(f"\n{name}:")
        print(f"  Input channels: {module.in_channels}")
        print(f"  Output channels: {module.out_channels}")
        print(f"  Kernel size: {module.kernel_size}")
        print(f"  Stride: {module.stride}")
        print(f"  Padding: {module.padding}")