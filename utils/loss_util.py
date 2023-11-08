import torch
import torch.nn.functional as F
import sys

sys.path.append("../")
from utils.AF.Fsmish import smish as Fsmish


class TEEDLoss:
    def __init__(self, loss_type='bdcn', device="cuda"):
        super(TEEDLoss, self).__init__()
        self.device = device
        self.loss_type = loss_type

        if loss_type == 'bdcn':
            self.loss_fn = self.bdcn_loss
        elif loss_type == 'cats':
            self.loss_fn = self.cats_loss

    def __call__(self, inputs, targets, reg=1):
        return self.loss_fn(inputs, targets, reg)

    def bdcn_loss_per_channel(self, inputs, targets, reg=1):
        # bdcn loss modified in DexiNed
        # targets = targets.long()
        # print(targets)
        mask = (targets != 0).float()
        num_positive = torch.sum((mask > 0.0)).float()  # >0.1
        num_negative = torch.sum((mask <= 0.0)).float()  # <=0.1

        mask[mask > 0.0] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
        mask[mask <= 0.0] = (
            1.1 * num_positive / (num_positive + num_negative)
        )  # before mask[mask <= 0.1]
        inputs = torch.sigmoid(inputs)
        cost = torch.nn.BCELoss(mask, reduction="none")(inputs, targets.float())
        cost = torch.sum(cost.float().mean((1, 2, 3)))  # before sum
        return reg * cost
    
    def bdrloss_per_channel(self, prediction, label, radius):
        """
        The boundary tracing loss that handles the confusing pixels.
        """

        filt = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1)
        filt.requires_grad = False
        filt = filt.to(self.device)

        bdr_pred = prediction * label
        pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

        texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
        mask = (texture_mask != 0).float()
        mask[label == 1] = 0
        pred_texture_sum = F.conv2d(
            prediction * (1 - label) * mask, filt, bias=None, stride=1, padding=radius
        )

        softmax_map = torch.clamp(
            pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10
        )
        cost = -label * torch.log(softmax_map)
        cost[label == 0] = 0

        return torch.sum(cost.float().mean((1, 2, 3)))


    def textureloss_per_channel(self, prediction, label, mask_radius):
        """
        The texture suppression loss that smooths the texture regions.
        """
        filt1 = torch.ones(1, 1, 3, 3)
        filt1.requires_grad = False
        filt1 = filt1.to(self.device)
        filt2 = torch.ones(1, 1, 2 * mask_radius + 1, 2 * mask_radius + 1)
        filt2.requires_grad = False
        filt2 = filt2.to(self.device)

        pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
        label_sums = F.conv2d(
            label.float(), filt2, bias=None, stride=1, padding=mask_radius
        )

        mask = 1 - torch.gt(label_sums, 0).float()

        loss = -torch.log(torch.clamp(1 - pred_sums / 9, 1e-10, 1 - 1e-10))
        loss[mask == 0] = 0

        return torch.sum(loss.float().mean((1, 2, 3)))


    def bdcn_loss(self, inputs, targets, reg=1):
        net_loss = 0
        for ch in range(inputs.shape[1]):
            net_loss += self.bdcn_loss_per_channel(inputs[:, ch:ch + 1, :, :], targets[:, ch:ch + 1, :, :], reg)
        net_loss /= inputs.shape[1]
        return net_loss
    
    def textureloss(self, prediction, label, mask_radius):
        net_loss = 0
        for ch in range(prediction.shape[1]):
            net_loss += self.textureloss_per_channel(prediction[:, ch:ch + 1, :, :], label[:, ch:ch + 1, :, :], mask_radius)
        net_loss /= prediction.shape[1]
        return net_loss

    def bdrloss(self, prediction, label, radius):
        net_loss = 0
        for ch in range(prediction.shape[1]):
            net_loss += self.bdrloss_per_channel(prediction[:, ch:ch + 1, :, :], label[:, ch:ch + 1, :, :], radius)
        net_loss /= prediction.shape[1]
        return net_loss
    
    def cats_loss(self, prediction, label, reg=[1, 1]):
        tex_factor, bdr_factor = reg
        balanced_w = 1.1
        label = label.float()
        prediction = prediction.float()
        with torch.no_grad():
            mask = label.clone()
            num_positive = torch.sum((mask == 1).float()).float()
            num_negative = torch.sum((mask == 0).float()).float()
            beta = num_negative / (num_positive + num_negative)
            mask[mask == 1] = beta
            mask[mask == 0] = balanced_w * (1 - beta)
            mask[mask == 2] = 0

        prediction = torch.sigmoid(prediction)

        cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(), label.float(), weight=mask, reduction="none"
        )
        cost = torch.sum(cost.float().mean((1, 2, 3))) 
        label_w = (label != 0).float()
        textcost = self.textureloss(
            prediction.float(), label_w.float(), mask_radius=4
        )
        bdrcost = self.bdrloss(prediction.float(), label_w.float(), radius=4)

        return cost + bdr_factor * bdrcost + tex_factor * textcost
