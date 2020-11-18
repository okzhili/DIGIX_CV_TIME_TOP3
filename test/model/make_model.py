import torch
import torch.nn as nn

from model.backbones.res2net import res2net101_v1b
from utils.pooling import GeM
from .backbones.resnest import resnest50, resnest101, resnest50_ibn, resnest101_ibn
from .backbones.ResNeSt.resnest_bdb import resnest101_bdb, resnest50_bdb
from .backbones.circle_loss import CircleLoss
from .backbones.mgn import MGN
from .backbones.osnet_ain import OSNet, osnet_ain_x1_0
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.resnext_ibn_a import resnext101_ibn_a, resnext50_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
import torch.nn.functional as F

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


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        self.name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'res2net101_normal':
            self.in_planes = 2048
            self.base = res2net101_v1b(last_stride)
            print('using res2net101_v1b as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnest101_ibn':
            self.in_planes = 2048
            self.base = resnest101_ibn(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name =='resnest':
            self.in_planes = 2048
            self.base = resnest50(last_stride=1)
        elif model_name == 'resnest_ibn':
            self.in_planes = 2048
            self.base = resnest50_ibn(last_stride=1)
        elif model_name == 'resnest101':
            self.in_planes = 2048
            self.base = resnest101(last_stride=1)
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'resnest101_bdb':
            self.in_planes = 2048
            self.base = resnest50_bdb(num_classes=1000, pretrained=True)
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride,frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'resnext':
            self.in_planes = 2048
            self.base = resnext101_ibn_a(1)
            print('using resnext as a backbone')
        elif model_name == 'b7':
            self.in_planes = 2560
            self.base = EfficientNet.from_pretrained('efficientnet-b7')
             # = nn.Sequential(*list(model.children())[:5])
            print('using b7 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            if model_name != 'mgn' and 'bdb' not in model_name:
                self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.pool = GeM()

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = self.pool(x).squeeze()
        feat = self.bottleneck(global_feat)
        return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'base.fc' in i or 'sub4' in i:
                continue
            if 'module' in i:
                self.state_dict()[i[7:]].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
