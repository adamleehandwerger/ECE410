import torch
import torchvision.models as models
from torchinfo import summary

# Load model
model = models.resnet18(pretrained=False)

# Get summary
stats = summary(model, input_size=(1, 3, 224, 224), verbose=0)

# Extract layer info
layers_info = []
for layer in stats.summary_list:
    if layer.is_leaf_layer and layer.macs > 0:
        layers_info.append({
            'name': layer.get_layer_name(True, True),
            'type': layer.class_name,
            'output_size': str(layer.output_size),
            'macs': layer.macs,
            'params': layer.num_params
        })

# Sort by MACs descending
layers_info.sort(key=lambda x: x['macs'], reverse=True)

# Get top 5 only
top_5 = layers_info[:5]

# Display top 5
print("="*100)
print("TOP 5 LAYERS BY MAC COUNT")
print("="*100)
print(f"{'Rank':<6} {'Layer Name':<35} {'MACs':<20} {'Scientific':<20} {'Parameters'}")
print("-"*100)

for i, info in enumerate(top_5, 1):
    print(f"{i:<6} {info['name']:<35} {info['macs']:>15,}   {info['macs']:>15.2e}   {info['params']:>15,}")

print("="*100)

# Detailed breakdown
print("\nDETAILED BREAKDOWN:")
print("="*100)

if not top_5:
    print("No layers found. Check that torchinfo is installed and macs are computed.")

for i, info in enumerate(top_5, 1):
    print(f"\n{i}. {info['name']}")
    print(f"   Type: {info['type']}")
    print(f"   MACs: {info['macs']:,}")
    print(f"   MACs (scientific): {info['macs']:.2e}")
    print(f"   Parameters: {info['params']:,}")
    print(f"   Output Size: {info['output_size']}")
