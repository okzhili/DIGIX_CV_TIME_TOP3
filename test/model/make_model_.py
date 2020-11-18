import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace
#from loss.circle_loss import CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.resnest import resnest50
from .backbones.res2net import res2net50_v1b,res2net101_v1b
from .backbones.resnext_ibn_a import resnext101_ibn_a
# from
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
# from .non_local import Non_local
# from .context_block import ContextBlock

## 必须使用该方法下载模型，然后加载
# from flyai.utils import remote_helper
# path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b3-5fb5a3c3.pth')


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BatchDrop(nn.Module):
    """ref: https://github.com/daizuozhuo/batch-dropblock-network/blob/master/models/networks.py
    batch drop mask
    """

    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):


        a = random.randint(0,30)
        # print(a)
        if a > 15:
            return x
        else:
            mask = x.new_ones(x.size())
            for i in range(a):
                if self.training:
                    # print("training")
                    h, w = x.size()[-2:]

                    rh = round(self.h_ratio * h)
                    rw = round(self.w_ratio * w)
                    sx = random.randint(0, h - rh)
                    sy = random.randint(0, w - rw)
                    # print(sx,sy,rh,rw)

                    mask[:, :, sx:sx + rh, sy:sy + rw] = 0
                    # mask[:, :, :,:] = 0
            x = x * mask
            # print("over")
        return x







# fpn
class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_name = cfg.MODEL.NAME
        self.num_classes = num_classes
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.dropblack = BatchDrop(0.05,0.05)

        # self.non_local = Non_local()
        # self.context_block = ContextBlock(2048,0.25)
        # self.context_block1 = ContextBlock(1024,0.25)


        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnest_512':
            self.in_planes = 2048
            self.base = resnest50(last_stride)
            print('using resnest50 as a backbone')
        elif model_name == 'res2net_512':
            self.in_planes = 2048
            self.base = res2net101_v1b(last_stride)
            print('using resnest50 as a backbone')
        elif model_name == 'res2net_320':
            self.in_planes = 2048
            self.base = res2net101_v1b(last_stride)
            print('using resnest50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnext101_ibn_a':
            self.in_planes = 2048
            self.base = resnext101_ibn_a(last_stride)
            print('using resnext101_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride,frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'res2net50_v1b':
            self.in_planes = 2048
            self.base = res2net50_v1b(last_stride)
            print('using res2net50_v1b as a backbone')
        elif model_name == 'res2net101_v1b':
            self.in_planes = 2048
            self.base = res2net101_v1b(last_stride)
            print('using res2net101_v1b as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = GeM()
        self.gap_sub1 = GeM()
        self.gap_sub2 = GeM()
        self.gap_sub3 = GeM()
        self.gap_sub4 = GeM()
        # self.conv5 = nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.conv5 = nn.Conv2d(2048,1024,kernel_size=1,bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        # self.bn5 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)


        #layer2
        # self.conv_layer2 = nn.Conv2d(1024,512,kernel_size=1,bias=False)
        # self.relu_layer2 = nn.ReLU(inplace=True)
        # self.gap_layer2 = GeM()
        # self.bottleneck_layer2 = nn.BatchNorm1d(512)
        # self.classifier_layer2 = nn.Linear(512, self.num_classes, bias=False)
        # self.classifier_layer2.apply(weights_init_classifier)


        # self.gap_sub4 = GeM()
        print("use Gem pooling")



        if model_name == 'EfficientNetb3':
            self.classifier = nn.Linear(1536, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        elif model_name == 'densenet169_ibn_a':
            self.classifier = nn.Linear(1664, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        elif model_name == 'densenet161':
            self.classifier = nn.Linear(2208, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)


            self.classifier_sub1 = nn.Linear(1024, self.num_classes, bias=False)
            self.classifier_sub1.apply(weights_init_classifier)
            #

            self.classifier_sub2 = nn.Linear(1024, self.num_classes, bias=False)
            self.classifier_sub2.apply(weights_init_classifier)
            #

            self.classifier_sub3 = nn.Linear(1024, self.num_classes, bias=False)
            self.classifier_sub3.apply(weights_init_classifier)
            #
            self.classifier_sub4 = nn.Linear(1024, self.num_classes, bias=False)
            self.classifier_sub4.apply(weights_init_classifier)





        if model_name == 'EfficientNetb3':
            self.bottleneck = nn.BatchNorm1d(1536)
        elif model_name == 'densenet169_ibn_a':
            self.bottleneck = nn.BatchNorm1d(1664)
        elif model_name == 'densenet161':
            self.bottleneck = nn.BatchNorm1d(2208)
        else:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            # self.bottleneck_sub1 = nn.BatchNorm1d(1024)
            # self.bottleneck_sub1 = nn.BatchNorm1d(256)
            # self.bottleneck_sub2 = nn.BatchNorm1d(256)
            # self.bottleneck_sub3 = nn.BatchNorm1d(256)
            # self.bottleneck_sub4 = nn.BatchNorm1d(256)
            self.bottleneck_sub1 = nn.BatchNorm1d(1024)
            self.bottleneck_sub2 = nn.BatchNorm1d(1024)
            self.bottleneck_sub3 = nn.BatchNorm1d(1024)
            self.bottleneck_sub4 = nn.BatchNorm1d(1024)


        # self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_sub1.bias.requires_grad_(False)
        self.bottleneck_sub1.apply(weights_init_kaiming)

        self.bottleneck_sub2.bias.requires_grad_(False)
        self.bottleneck_sub2.apply(weights_init_kaiming)
        #
        #
        self.bottleneck_sub3.bias.requires_grad_(False)
        self.bottleneck_sub3.apply(weights_init_kaiming)

        # self.bottleneck_sub4.bias.requires_grad_(False)
        # self.bottleneck_sub4.apply(weights_init_kaiming)


    def upsample(self,x):
        _,_,H,W = x.size()
        return F.upsample(x, size=(2*H, 2*W), mode='bilinear')


    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        # x = self.base(x)

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
       # x = self.dropblack(x)



        x = self.base.layer3(x)
        #x = self.dropblack(x)
        x1 = x[:,:,:16,:]
        x2 = x[:,:,16:,:]
        x3 = x[:,:,:,:]

        # print(x.shape)
        #x3与x4融合
        x4 = self.base.layer4(x)
        x4 = self.dropblack(x4)

        sub_feat1 = self.gap_sub1(x1)

        sub_global_feat1 = sub_feat1.flatten(1)
        sub_bn1 = self.bottleneck_sub1(sub_global_feat1)

        sub_feat2 = self.gap_sub2(x2)
        sub_global_feat2 = sub_feat2.flatten(1)
        sub_bn2 = self.bottleneck_sub2(sub_global_feat2)
        #
        #
        sub_feat3 = self.gap_sub3(x3)
        sub_global_feat3 = sub_feat3.flatten(1)
        # sub_global_feat3 = sub_global_feat3.clamp(max=1)
        sub_bn3 = self.bottleneck_sub3(sub_global_feat3)
        #



        # print(x.shape[2:4])
        # global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        global_feat = self.gap(x4)
        global_feat = global_feat.flatten(1)


        # global_feat = global_feat.clamp(max=1)
        feat = self.bottleneck(global_feat)
        # global_cat = torch.cat((global_feat, sub_global_feat1, sub_global_feat2), 1)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
                score1 = self.classifier_sub1(sub_bn1,label)
                score2 = self.classifier_sub2(sub_bn2,label)
            else:
                cls_score = self.classifier(feat)
                score1 = self.classifier_sub1(sub_bn1)
                score2 = self.classifier_sub2(sub_bn2)
                # score4 = self.classifier_sub4(sub_bn4)
            # return cls_score, global_feat  # global feature for triplet loss
            # return cls_score, global_feat ,score1,score2 # global feature for triplet loss
            return cls_score, global_feat ,score1,score2,sub_bn3 # global feature for triplet loss
            # return cls_score, global_feat ,score1,score2,sub_bn3,sub_global_feat4,score4 # global feature for triplet loss
            # return cls_score, global_feat ,score1 # global feature for triplet loss
            # return cls_score, global_cat ,score1,score2 # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("after")
                # print("Test with feature after BN")

                # cat = torch.cat((feat,sub_bn3,sub_bn4),1)
                cat = torch.cat((feat,sub_bn3),1)
                # cat = torch.cat((global_feat,sub_global_feat3),1)
                # print(cat.shape,feat.shape)

                return cat
                # return feat
            else:
                # print("Test with feature before BN")
                return global_feat




    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'base.fc' in i or 'sub4' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model_(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
