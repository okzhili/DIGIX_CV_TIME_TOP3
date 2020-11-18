# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss_scr,TripletLoss_xbm
from .center_loss import CenterLoss
from .rank_loss import RankedLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss_scr()
            xbm = TripletLoss_xbm()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat,target,sub_feat1,sub_feat2,feat1):
        # def loss_func(score, feat,target,sub_feat1,sub_feat2,feat1,score_layer2):
        # def loss_func(score, feat,target,sub_feat1,sub_feat2,feat1,feat4,score4):
            # print(sub_feat2.shape,sub_feat3.shape)
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # print('using right sampler and loss')
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    if feat1!=None:
                        return 0.5*F.cross_entropy(score, target)+  +0.5/2*(F.cross_entropy(sub_feat1, target)+ \
                            F.cross_entropy(sub_feat2, target))+ 0.25*(triplet(feat, target)[0]+xbm(feat, target)[0])\
                               +0.25*(triplet(feat1, target)[0]+xbm(feat1, target)[0])
                    else:
                        # print("F.cross_entropy(score, target)",F.cross_entropy(score, target))
                        # print("triplet(feat, target)[0]",triplet(feat, target)[0])
                        # print("xbm(feat, target)[0]",xbm(feat, target)[0])
                        xbm_loss = xbm(feat, target)[0]
                        if xbm_loss == float('inf'):
                            print("xbm inf")
                            return F.cross_entropy(score, target) +  triplet(feat, target)[0]
                        else:
                            return F.cross_entropy(score, target) + 0.5 * triplet(feat, target)[0] + 0.5 * \
                                   xbm_loss

                        # print('layer4训练')
                    # return 0.5*F.cross_entropy(score_layer2, target)+ 0.5*F.cross_entropy(score, target)+  +0.5/2*(F.cross_entropy(sub_feat1, target)+F.cross_entropy(sub_feat2, target))+ 0.5*triplet(feat, target)[0]+0.5*triplet(feat1, target)[0]
                    #     return 1*F.cross_entropy(score, target)+ 1*triplet(feat, target)[0]
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'rankloss':
                print('using right sampler and loss')
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    print("use rank loss")
                    return 0.5 * F.cross_entropy(score, target) + 0.25 * (
                                F.cross_entropy(sub_feat1, target) + F.cross_entropy(sub_feat2, target)) + \
                           RankedLoss()(feat, target)[0]

            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet_center':
        def loss_func(score, feat,sub_feat1,sub_feat2, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                           triplet(feat, target)[0] + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + \
                           triplet(feat, target)[0] + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


