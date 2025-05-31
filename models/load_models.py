from robustbench.utils import load_model as load_robust_model
from torchvision import models
import torch.nn as nn
import torch
import os
# Optional: import custom ResNet56 if you have it
# from your_models.resnet56 import resnet56

def load_victim_model(model_info, config):
    """
    Loads a model based on model_info dict and config.
    Supports both robustbench and standard models.

    Args:
        model_info (dict): e.g., {"name": "Wong2020Fast", "type": "robust"}
        config (dict): general configuration

    Returns:
        model (torch.nn.Module)
    """
    custom_cache = config.get("robustbench_cache", os.path.expanduser("~/.robustbench"))
    name = model_info["name"]
    model_type = model_info["type"]

    if model_type == "robust":
        model = load_robust_model(
            model_name=name,
            dataset=config.get("dataset", "cifar10"),
            threat_model=config.get("threat_model", "Linf"),
            model_dir=custom_cache
        )

    elif model_type == "normal":
        if name == "VGG19":
            model = models.vgg19(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        elif name == "MobileNetV2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        elif name == "ShuffleNetV2":
            model = models.shufflenet_v2_x1_0(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif name == "ResNet56":
            raise NotImplementedError("ResNet56 loading not implemented. Add custom import.")
            # model = resnet56()
            # model.load_state_dict(torch.load(f"./pretrained/ResNet56.pth"))
        else:
            raise ValueError(f"Unknown model name: {name}")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model
