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
            if bmask:
                loss = lambda_ * weight * torch.mean(mask.reshape((-1, 1)) * (output - target) ** 2 * mask)
            else:
                loss = lambda_ * weight * torch.mean((output - target) ** 2)
        else:
            if bmask:
                loss = lambda_ * weight * torch.mean(mask.reshape((-1, 1)) * ((output - target) ** 2 + (spec_recons - target) ** 2) * mask)
            else:
                loss = lambda_ * weight * torch.mean((output - target) ** 2 + (spec_recons - target) ** 2)




    #now compute the loss only on the postive edges
    #binarize the target. We need it because for weighted network we donot have (0,1) values.

    elif loss_type=='sep':
        # t1=time.time()
        pos_indices = torch.where(target>0, 1, 0)
        neg_indices = torch.where(target<=0, 1, 0)
        # print(time.time()-t1)

        n_pos = torch.count_nonzero(pos_indices)
        n_neg = torch.count_nonzero(neg_indices)

        out = (output - target) ** 2
        pos_loss = torch.mul(pos_indices, out)
        neg_loss = torch.mul(neg_indices, out)
        # l = lambda_ * weight * torch.mean(mask.reshape((-1, 1))*(pos_loss + neg_loss)*mask)
        # assert (loss-l)<0.00001, print('not equal')

        #find average loss over pos and neg edges separately
        mean_pos_loss = torch.sum(pos_loss)/n_pos
        mean_neg_loss = torch.sum(neg_loss)/n_neg

        loss = mean_neg_loss + mean_pos_loss
    return loss


cls_criterion = nn.BCEWithLogitsLoss(reduction="none")


def classification_loss(output: Tensor, target: Tensor, mask: Tensor, lambda_: float) -> Tensor:
    """Multi-label classification loss used when labels are provided.
    """

    loss = (1 - lambda_) * (mask.reshape((-1, 1)) * cls_criterion(output, target)).mean()
    return loss
