import torch
import torch.nn as nn
import torch.nn.functional as F


# CenterNet各部分所对应的loss类


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).byte()
            loss = F.binary_cross_entropy(pred.masked_select(mask),
                                          target.masked_select(mask),
                                          reduction='mean')
            return loss
        else:
            return 0.


class OffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(OffSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).byte()
            loss = F.smooth_l1_loss(pred.masked_select(mask),
                                    target.masked_select(mask),
                                    reduction='mean')
            return loss
        else:
            return 0.


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def isnan(x):
    return x != x


class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_wh = OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()

    def forward(self, pr_decs, gt_batch):
        # 计算最浅层的loss
        hm_loss_P2 = self.L_hm(pr_decs['hm_P2'], gt_batch['hm_P2'])
        wh_loss_P2 = self.L_wh(pr_decs['wh_P2'], gt_batch['reg_mask_P2'], gt_batch['ind_P2'], gt_batch['wh_P2'])
        off_loss_P2 = self.L_off(pr_decs['reg_P2'], gt_batch['reg_mask_P2'], gt_batch['ind_P2'], gt_batch['reg_P2'])
        cls_theta_loss_P2 = self.L_cls_theta(pr_decs['cls_theta_P2'], gt_batch['reg_mask_P2'], gt_batch['ind_P2'],
                                             gt_batch['cls_theta_P2'])
        if isnan(hm_loss_P2) or isnan(wh_loss_P2) or isnan(off_loss_P2):
            print('hm loss is {}'.format(hm_loss_P2))
            print('wh loss is {}'.format(wh_loss_P2))
            print('off loss is {}'.format(off_loss_P2))
        loss = 0.8*hm_loss_P2 + wh_loss_P2 + off_loss_P2 + cls_theta_loss_P2
        return loss