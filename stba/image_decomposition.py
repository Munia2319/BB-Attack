import torch
import torch.nn.functional as F

def decompose_image(x, kernel_size=3, sigma=1.0):
    """
    Decomposes input image into high-frequency and low-frequency parts using Gaussian blur.

    Args:
        x (Tensor): Input image of shape [B, C, H, W]
        kernel_size (int): Kernel size for Gaussian blur (default: 3)
        sigma (float): Standard deviation of the Gaussian kernel (default: 1.0)

    Returns:
        x_low (Tensor): Low-frequency (blurred) version of x
        x_high (Tensor): High-frequency details (x - x_low)
    """
    padding = kernel_size // 2
    channels = x.shape[1]

    # Create a Gaussian kernel
    def get_gaussian_kernel(k, sigma):
        ax = torch.arange(-k // 2 + 1., k // 2 + 1.)
        xx, yy = torch.meshgrid([ax, ax], indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    kernel = get_gaussian_kernel(kernel_size, sigma).to(x.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    x_low = F.conv2d(x, kernel, padding=padding, groups=channels)
    x_high = x - x_low

    return x_low, x_high
