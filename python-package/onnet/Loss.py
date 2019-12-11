import torch
import torch.nn.functional as F

class UserLoss(object):

    @staticmethod
    def cys_loss(output, target, reduction='mean'):
        #loss = F.binary_cross_entropy(output, target, reduction=reduction)
        loss = F.cross_entropy(output, target, reduction=reduction)
        #loss = F.nll_loss(output, target, reduction=reduction)

        return loss