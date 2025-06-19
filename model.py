import torch
import torch.nn as nn
import sys
import os
from torchvision import models
from robustbench.utils import load_model as load_robust_model
from robustbench.model_zoo.enums import ThreatModel

# Import CIFAR and STL-10 model definitions
from models.cifar_models import (
    cifar10_mobilenetv2_x0_5, cifar10_resnet20, cifar10_resnet56, cifar10_shufflenetv2_x2_0,
    cifar10_vgg16_bn, cifar10_vgg19_bn, cifar10_vit_b16, cifar10_repvgg_a0,
    cifar100_mobilenetv2_x0_5, cifar100_resnet20, cifar100_resnet56, cifar100_shufflenetv2_x2_0,
    cifar100_vgg16_bn, cifar100_vgg19_bn, cifar100_vit_b16, cifar100_repvgg_a0
)
from models.pytorch_cifar_models import DenseNet121 as cifar10_densenet121
from models.pytorch_cifar_models import EfficientNetB0 as cifar10_efficientnetb0

BASE_PATH = "/home/mrliu/work/TEST_CODE/pytorch-cifar/checkpoint/"


def normalize_fn(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return f'mean={self.mean}, std={self.std}'


def load_clean_model(args, device=None):
    if args.target_model == "defense":
        return load_defended_model(args, device)

    if args.dataset == "CIFAR-10":
        if args.model == 'vgg16_bn':
            net = cifar10_vgg16_bn(pretrained=True)
        elif args.model == 'vgg19_bn':
            net = cifar10_vgg19_bn(pretrained=True)
        elif args.model == "resnet56":
            net = cifar10_resnet56(pretrained=True)
        elif args.model == "resnet20":
            net = cifar10_resnet20(pretrained=True)
        elif args.model == "mobilenetv2":
            net = cifar10_mobilenetv2_x0_5(pretrained=True)
        elif args.model == "shufflenetv2":
            net = cifar10_shufflenetv2_x2_0(pretrained=True)
        elif args.model == "repvgg":
            net = cifar10_repvgg_a0(pretrained=True)
        elif args.model == "vit_b16":
            net = cifar10_vit_b16(pretrained=True)
        elif args.model == "densenet121":
            net = cifar10_densenet121()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/kpt-densenet121.pth')
            net.load_state_dict(checkpoint['net'])
        elif args.model == "efficientnetb0":
            net = cifar10_efficientnetb0()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/ckpt-efficientnetb0.pth')
            net.load_state_dict(checkpoint['net'])
        else:
            print("Not implemented!!!", args.model)
            sys.exit(1)

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        # Normalize the model input using the channel-wise mean and std
        normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        # Wrap the model with normalization
        net = nn.Sequential(normalize, net)

    elif args.dataset == "CIFAR-100":
        if args.model == 'vgg16_bn':
            net = cifar100_vgg16_bn(pretrained=True)
        elif args.model == 'vgg19_bn':
            net = cifar100_vgg19_bn(pretrained=True)
        elif args.model == "resnet56":
            net = cifar100_resnet56(pretrained=True)
        elif args.model == "resnet20":
            net = cifar100_resnet20(pretrained=True)
        elif args.model == "mobilenetv2":
            net = cifar100_mobilenetv2_x0_5(pretrained=True)
        elif args.model == "shufflenetv2":
            net = cifar100_shufflenetv2_x2_0(pretrained=True)
        elif args.model == "repvgg":
            net = cifar100_repvgg_a0(pretrained=True)
        elif args.model == "vit_b16":
            net = cifar100_vit_b16(pretrained=True)
        elif args.model == "densenet121":
            net = cifar10_densenet121()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/ckpt-densenet121-cifar100.pth')
            net.load_state_dict(checkpoint['net'])
        elif args.model == "efficientnetb0":
            net = cifar10_efficientnetb0()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(f'{BASE_PATH}/ckpt-efficientnetb0-cifar100.pth')
            net.load_state_dict(checkpoint['net'])
        else:
            print("Not implemented!!!", args.model)
            sys.exit(1)

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        net = nn.Sequential(normalize, net)

    elif args.dataset in ["ImageNet", "NIPS2017"]:
        if args.model == 'inceptionv3':
            net = models.inception_v3(pretrained=True)
        elif args.model == 'vgg16':
            net = models.vgg16(pretrained=True)
        elif args.model == 'vgg19':
            net = models.vgg19(pretrained=True)
        elif args.model == "resnet50":
            net = models.resnet50(pretrained=True)
        elif args.model == "resnet152":
            net = models.resnet152(pretrained=True)
        elif args.model == "densenet121":
            net = models.densenet121(pretrained=True)
        elif args.model == "wide_resnet50":
            net = models.wide_resnet50_2(pretrained=True)
        elif args.model == "shufflenetv2":
            net = models.shufflenet_v2_x0_5(pretrained=True)
        elif args.model == "mobilenetv2":
            net = models.mobilenet_v2(pretrained=True)
        else:
            print("Not implemented!!!", args.model)
            sys.exit(1)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        net = nn.Sequential(normalize, net)

    else:
        print("Not implemented!!!")
        sys.exit(1)

    net.eval().to(device)
    return net

def load_defended_model(args, device=None):
    model_name = args.model
    try:
        print(f"[INFO] Loading defended model: {model_name}")
        model = load_robust_model(
            model_name=model_name,
            dataset='cifar10',
            threat_model=ThreatModel.Linf
        )
        model.eval().to(device)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load defended model {model_name}: {e}")
        sys.exit(1)
