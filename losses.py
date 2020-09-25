import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss


class CrossEntropy(_WeightedLoss):
    
    def __init__(self, weight = None):

        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight = weight)

        return

    def forward(self, pred, target):

        loss = self.criterion(pred, target)

        return loss


class DiceLoss(_WeightedLoss):

    def __init__(self, eps = 1e-6):

        super(DiceLoss, self).__init__()
        self.eps = eps
        self.softmax = nn.Softmax(dim = 1)

        return

    def forward(self, pred, target):

        pred = self.softmax(pred)
        target = (pred.detach() * 0).scatter_(1, target.unsqueeze(1), 1)

        numerator = 2 * (pred * target).sum(dim = (2, 3, 4)) + self.eps
        denominator = (pred + target).sum(dim = (2, 3, 4)) + self.eps
        loss_per_channel = (1 - numerator / denominator)
        loss = loss_per_channel.sum() / pred.shape[0] / pred.shape[1]

        return loss


class CombinedLoss(_Loss):
    
    def __init__(self):

        super(CombinedLoss, self).__init__()
        self.loss0 = CrossEntropy()
        self.loss1 = DiceLoss()

        return 

    def forward(self, pred, target):

        loss = self.loss0(pred, target) + self.loss1(pred, target)

        return loss
