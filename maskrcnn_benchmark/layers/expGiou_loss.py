import torch
from torch import nn


class ExpGIOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        C_left = torch.min(pred_left,target_left)
        C_top = torch.min(pred_top,target_top)
        C_right = torch.max(pred_right,target_right)
        C_bottom = torch.max(pred_bottom,target_bottom)

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        C_aera = (C_left + C_right) * \
                (C_top + C_bottom )

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        #giou 
        giou = (area_intersect) / (area_union) - (C_aera - area_union)/(C_aera)
        # losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        # giou losses
        # losses = 1 - giou
        # giou exp loss 
        losses = (1-giou)*torch.exp(-giou)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()
