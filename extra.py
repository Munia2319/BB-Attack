import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# Path where STBA expects the batchfile for defended models
save_path = "data/CIFAR-10_Sehwag2020Hydra_Wang2020Improving_Zhang2019Theoretically_Wong2020Fast_robust.pth"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Transform: convert images to tensor only (no normalization)
transform = transforms.ToTensor()

# Load CIFAR-10 test set
testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)

# Fetch all test images & labels
images, labels = next(iter(testloader))

# Save to .pth file
torch.save({'images': images, 'labels': labels}, save_path)

print(f"✅ Saved CIFAR-10 defense batchfile to: {save_path}")
