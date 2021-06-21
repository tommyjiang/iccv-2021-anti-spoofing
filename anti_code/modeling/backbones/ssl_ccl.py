import torch
from torch import nn

from .resnet import ResNet, BasicBlock, Bottleneck
from .resnet_ibn_a import resnet50_ibn_a
from .efficient import EfficientNetAntiSpoof
from .vit import ViT
from .swin import SwinTransformer
from .senet import SENet,SEResNetBottleneck,SEResNeXtBottleneck
from .resnet_ibn_a import resnet101_ibn_a
from .ssl_resnet import build_ResNet


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

class SSL_CCL(nn.Module):
    in_planes = 2048
    def __init__(self,num_class=2,encoder_name = 'resnet50',last_strider=2):
        super(SSL_CCL,self).__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_class
        if encoder_name == 'resnet50':
            self.base = build_ResNet(50,num_class)
        elif encoder_name == 'resnet101':
            self.base = build_ResNet(101,num_class)
        elif encoder_name == 'resnet152':
            self.base = build_ResNet(152,num_class)
        elif encoder_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=2)
        ## Projector & Predictor
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_planes, self.num_classes)
        self.classifier_3 = nn.Linear(self.in_planes, 3)
        self.projector = nn.Sequential(nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512,128))
        self.predictor = nn.Sequential(nn.Linear(128,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512,128))
        # self.classifier_3 = nn.Linear(self.)
    
    def forward_once(self,x,flag=False):
        output_list = [self.base(x)]
        x = output_list[-1]
        # print(x.shape)
        feat = self.gap(x)
    # feat_pr = feat
        feat = feat.view(feat.shape[0], -1)
        output_list.append(feat)
        # print(feat.shape)
        
        score = self.classifier(feat)
        output_list.append(score)
        score_0 = self.classifier_3(feat)
        output_list.append(score_0)
        # print(score.shape)
        if not self.training:
            return output_list,feat
        feat_proj = self.projector(feat)
        if flag:
            # feat_proj = self.projector(feat)
            return output_list,feat_proj
        # print(feat_proj.shape)
        feat_pred = self.predictor(feat_proj)

        return output_list, feat_pred
    def forward(self,x1,x2=None,return_fmap=False,flag=False):
        if x2==None:
            outlist0, feat0 = self.forward_once(x1,flag)
        else:
            outlist0, feat0 = self.forward_once(x1,flag)
            outlist1, feat1 = self.forward_once(x2,flag)
        ## 只用一张图做无监督学习，两张图做ccl
        if return_fmap:
            return outlist0[:-1]
        else:
            return outlist0[-2],outlist0[-1],feat0,outlist1[-2],outlist1[-1],feat1

        # score1,score_mask1,feat1 = self.forward_once(x1,flag=exp)
        # score2,score_mask2,feat2 = self.forward_once(x2,flag=exp)
        # if not self.training:
        #     return score1
        # return score1,score_mask1,feat1,score2,score_mask2,feat2
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        # for j in self.state_dict():
        #     print(j)
        for i in param_dict:
            # print(i)
            # print(i)
            if 'module' in i:
                j = i.replace('module.','',1)
            else:
                j = i
            self.state_dict()[j].copy_(param_dict[i])
    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag        



    
