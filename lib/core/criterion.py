# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config

import numpy as np
from utils.utils import get_confusion_matrix



def get_region_cls_gt(gt_size, prob, label):
    # print(gt_size)
    gt = torch.ones((gt_size[0] ,gt_size[2], gt_size[3]))  #  2 * 32 * 64
    pred  = prob
    size = label.size()
    k_h = 32
    k_w = 64
    for n in range(gt_size[0]):
        for r in range(k_h):
            for c in range(k_w):
                child_label = label[n,c * size[-2] // k_h: (c+1) * size[-2] // k_h, r * size[-1] // k_w :(r+1) * size[-1] // k_w].unsqueeze(0)
                child_pred = pred[n,:, c * size[-2] // k_h: (c+1) * size[-2] // k_h, r * size[-1] // k_w :(r+1) * size[-1] // k_w].unsqueeze(0)
                child_confusion_matrix=get_confusion_matrix(
                    child_label,
                    child_pred,
                    child_label.size(),
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )
                # print(child_confusion_matrix)
                child_pos = child_confusion_matrix.sum(1)
                child_res = child_confusion_matrix.sum(0)
                child_tp = np.diag(child_confusion_matrix)
                # print(child_pos, child_res, child_tp)
                
                child_pixel_acc = child_tp.sum()/child_pos.sum()
                if child_pos.sum() == 0:
                    child_pixel_acc = 1.0
                # print('child_pixel_acc', child_pixel_acc)
                child_mean_acc = (child_tp/ np.maximum(1.0, child_pos)).mean()
                child_IoU_array = (child_tp / np.maximum(1.0, child_pos + child_res - child_tp))
                child_mean_IoU = child_IoU_array.mean()
                if child_mean_acc >= 0.95:
                    gt[n, r, c] = 1
                else:
                    gt[n, r, c] = 0
    return gt
    
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def region_cls_loss(self, feat, score, target):
        # feat  N * 2 * 32 * 64
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        # gt = torch.ones((2 ,feat.size(2), feat.size(3))).long().cuda(score.device)
        gt_size = feat.size()
        gt = get_region_cls_gt(gt_size, score, target)
    
        gt = gt.long().cuda(score.device)
        
        loss = self.loss_fn(feat,gt)
        return loss

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):
        loss_cls = None
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]
        if config.MODEL.NUM_OUTPUTS == 3:
            feat = score[2]
            score = score[:2]
            loss_cls = self.region_cls_loss(feat, score[0], target)
        
        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)
        loss_seg = sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])
        if loss_cls == None:
            all_loss = loss_seg
        else:
            all_loss = loss_seg + loss_cls
        return all_loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])
