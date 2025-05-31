import torch
import torch.nn.functional as F

def apply_flow_to_high_freq(x_high, flow, mode='bilinear', padding_mode='border', align_corners=True):
    """
    Apply a spatial flow field to the high-frequency part of an image using bilinear interpolation.

    Args:
        x_high (Tensor): [B, C, H, W] high-frequency image
        flow (Tensor): [B, 2, H, W] flow field: Δu (horizontal) and Δv (vertical)
        mode (str): Interpolation mode: 'bilinear' or 'nearest'
        padding_mode (str): Padding mode: 'zeros', 'border', or 'reflection'
        align_corners (bool): Whether to align corners in grid_sample

    Returns:
        x_transformed (Tensor): [B, C, H, W] spatially transformed image
    """
    B, C, H, W = x_high.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x_high.device),
        torch.linspace(-1, 1, W, device=x_high.device),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

    norm_flow = torch.zeros_like(flow)
    norm_flow[:, 0, :, :] = flow[:, 0, :, :] / (W / 2)
    norm_flow[:, 1, :, :] = flow[:, 1, :, :] / (H / 2)

    grid = base_grid + norm_flow
    grid = grid.permute(0, 2, 3, 1)

    x_transformed = F.grid_sample(
        x_high,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    return x_transformed
