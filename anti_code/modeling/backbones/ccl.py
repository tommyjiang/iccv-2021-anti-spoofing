import torch
from torch import nn

from .resnet import ResNet, BasicBlock, Bottleneck
from .resnet_ibn_a import resnet50_ibn_a
from .efficient import EfficientNetAntiSpoof
from .vit import ViT
from .swin import SwinTransformer
from .senet import SENet,SEResNetBottleneck,SEResNeXtBottleneck
from .resnet_ibn_a import resnet101_ibn_a


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

class CCL(nn.Module):
    in_planes = 2048
    def __init__(self,num_class=2,encoder_name = 'resnet50',last_strider=2):
        super(CCL,self).__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_class
        if encoder_name == 'resnet50':
            self.base = ResNet(last_stride=last_strider,block = Bottleneck,layers=[3,4,6,3])
        elif encoder_name == 'resnet101':
            self.base = ResNet(last_stride=last_strider,block = Bottleneck,layers=[3,4,23,3])
        elif encoder_name == 'resnet152':
            self.base = ResNet(last_stride=last_strider,block = Bottleneck,layers=[3,8,36,3])
        elif encoder_name =='efficient_b0':
            self.base = EfficientNetAntiSpoof(arch="b0").base
        elif encoder_name =='efficient_b5':
            self.base = EfficientNetAntiSpoof(arch="b5").base
        elif encoder_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_strider)
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
        elif encoder_name == 'resnet101_ibn_a':
            self.base = resnet101_ibn_a(last_stride=2)
        elif encoder_name == 'vit':
            self.base = ViT(
                image_size = 224,
                patch_size = 32,
                num_classes = 2,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        elif encoder_name == 'swin':
            self.base = SwinTransformer(num_classes=self.num_classes)
        else:
            print("not support "+ encoder_name)
        

        ## Projector & Predictor
        self.gap = nn.AdaptiveAvgPool2d(1)
        if self.encoder_name == 'efficient_b0':
            self.classifier = nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)
            self.classifier_3 = nn.Linear(1280, 3)
            self.projector = nn.Sequential(nn.Linear(1280,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512,128))
        elif self.encoder_name == 'efficient_b5':
            self.classifier = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
            self.classifier_3 = nn.Linear(2048, 3)
            self.projector = nn.Sequential(nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512,128))
        elif self.encoder_name == 'vit':
            self.classifier = nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024, self.num_classes))
            self.classifier_3 = nn.Linear(1024, 3)
            self.projector = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512,128))
        elif self.encoder_name == 'swin':
            self.classifier = nn.Linear(in_features=768,out_features=self.num_classes,bias=True)
            self.classifier_3 = nn.Linear(768, 3)
            self.projector = nn.Sequential(nn.Linear(768,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512,128))
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier_3 = nn.Linear(self.in_planes, 3)
            self.projector = nn.Sequential(nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512,128))

        self.predictor = nn.Sequential(nn.Linear(128,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512,128))
        # self.feat_reduce = nn.Sequential(nn.Linear(2048,128),nn.BatchNorm1d(128))
        # self.classifier_3 = nn.Linear(self.)
    
    def forward_once(self,x,flag=False):
        if 'b0' in self.encoder_name or 'b5' in self.encoder_name:
            x = self.base.extract_features(x)
        else:
            x = self.base(x)
        # print(x.shape)
        if self.encoder_name not in ['vit', 'swin']:
            feat = self.gap(x)
        # feat_pr = feat
            feat = feat.view(feat.shape[0], -1)
        else:
            feat = x
        # print(feat.shape)
        # feat_center = self.reduce
        
        score = self.classifier(feat)
        score_0 = self.classifier_3(feat)
        # print(score.shape)
        ## 主要是在推理的时候不用继续往后计算
        if not self.training:
            return score,score_0,feat,feat
        feat_proj = self.projector(feat)
        ## 区分正常模型和滑动平均模型
        if flag:
            # feat_proj = self.projector(feat)
            return score,score_0,feat_proj,feat
        # print(feat_proj.shape)
        feat_pred = self.predictor(feat_proj)

        return score,score_0, feat_pred, feat
    

    def forward(self,x1,x2,exp=False, isPair=True):
        score1,score_mask1,feat1,feat_c1 = self.forward_once(x1,flag=exp)
        if not self.training:
            return score1
        if not isPair:
            return score1,score_mask1,feat1,feat_c1
        score2,score_mask2,feat2,feat_c2 = self.forward_once(x2,flag=exp)
        return score1,score_mask1,feat1,score2,score_mask2,feat2,feat_c1,feat_c2

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



    
