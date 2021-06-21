# encoding: utf-8
import torch.nn.functional as F
import torch

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth, ContrastiveLoss
from .center_loss import CenterLoss
from .arch_face import ArcFace,CircleLoss
from .focal_loss import focal_loss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    # if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
    #     triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    # else:
    #     print('expected METRIC_LOSS_TYPE should be triplet'
    #           'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    else:
        xent = torch.nn.CrossEntropyLoss()

    if sampler == 'softmax':
        def loss_func(score, target):
            return xent(score, target)
    return loss_func

def make_ccl_loss(cfg, num_classes):
    # if cfg.MODEL.SOFTMAX_TYPE == 'focal'
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    elif cfg.MODEL.IF_LABELSMOOTH == 'focal':
        xent = focal_loss(alpha=0.5,gamma=2,num_classes=num_classes)
    else:
        xent = torch.nn.CrossEntropyLoss()
    constrtive = ContrastiveLoss()
    def loss_func(score0,target0,score1,target1):
        return (xent(score0,target0)+xent(score1,target1)) / 2
    def loss_constra(feat0,feat1,pair_label):
        return constrtive(feat0,feat1,pair_label)
    
    return loss_func,loss_constra

