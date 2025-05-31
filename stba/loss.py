import torch
import torch.nn.functional as F

def margin_loss(logits, labels):
    """
    Margin-based adversarial loss from Equation (6) of the STBA paper:
        L_adv = max( C(x_adv)_y - max_{k ≠ y} C(x_adv)_k, 0 )
    
    Args:
        logits (Tensor): Logits from model, shape [B, num_classes]
        labels (Tensor): True labels, shape [B]
    
    Returns:
        loss (Tensor): Scalar tensor representing margin loss
    """
    B = logits.shape[0]

    # Logit for the correct class
    correct_logit = logits[torch.arange(B), labels]

    # Mask out the correct class and find max logit of incorrect classes
    inf_mask = torch.ones_like(logits) * float('-inf')
    inf_mask[torch.arange(B), labels] = 0
    wrong_logits = logits + inf_mask
    max_wrong_logit, _ = wrong_logits.max(dim=1)

    margin = correct_logit - max_wrong_logit
    return margin.clamp(min=0).mean()


def compute_total_loss(logits, label, flow, lambda_, targeted=False):
    """
    Compute the total STBA loss:
        L_total = L_adv + lambda * L_flow
    
    Args:
        logits (Tensor): Model output logits, shape [B, num_classes]
        label (Tensor): Ground-truth or target label, shape [B]
        flow (Tensor): Flow field, shape [B, 2, H, W]
        lambda_ (float): Weight for flow smoothness loss
        targeted (bool): True for targeted attack (uses cross-entropy), False for untargeted (uses margin loss)

    Returns:
        total_loss (Tensor): Combined total loss
        loss_adv (Tensor): Adversarial component
        loss_flow (Tensor): Flow smoothness component
    """

    if targeted:
        # Targeted attack uses standard cross-entropy
        loss_adv = F.cross_entropy(logits, label)
    else:
        # Untargeted STBA uses margin loss from Eq. (6)
        loss_adv = margin_loss(logits, label)

    # === Flow Smoothness Loss (Eq. 7) ===
    u = flow[:, 0, :, :].unsqueeze(1)  # Horizontal flow
    v = flow[:, 1, :, :].unsqueeze(1) # Vertical flow

    du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]
    dv_dx = v[:, :, :, 1:] - v[:, :, :, :-1]
    du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]
    dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]

    eps = 1e-6  # To avoid sqrt(0)
    loss_dx = torch.sqrt(du_dx[:, :, :, :-1]**2 + dv_dx[:, :, :, :-1]**2 + eps).mean()
    loss_dy = torch.sqrt(du_dy[:, :, :-1, :]**2 + dv_dy[:, :, :-1, :]**2 + eps).mean()

    loss_flow = loss_dx + loss_dy

    total_loss = loss_adv + lambda_ * loss_flow
    return total_loss, loss_adv.detach(), loss_flow.detach()




