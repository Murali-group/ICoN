import torch
import torch.nn as nn
from torch import Tensor

from utils.common import Device
import time

def masked_scaled_mse(
    output: Tensor,
    target: Tensor,
    weight: Tensor,
    node_ids: Tensor,
    mask: Tensor,
    lambda_: float,
    device=None,
    loss_type=None,
    bmask=True,
    spec_recons=None
) -> Tensor:
    """Masked and scaled MSE loss.
    """

    if device is None:
        device = Device()

    # Subset `target` to current batch and make dense
    target = target.to(device)
    target = target.adj_t[node_ids, node_ids].to_dense()

    if loss_type == 'single':
        if spec_recons is None:
            loss = lambda_ * weight * torch.mean((output - target) ** 2)
        else:
            loss = lambda_ * weight * torch.mean((output - target) ** 2 + (spec_recons - target) ** 2)

    return loss
