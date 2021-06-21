import torch.nn as nn 
from efficientnet_pytorch import EfficientNet
import torch

class EfficientNetAntiSpoof(nn.Module):
    def __init__(self, arch="b5"):
        super(EfficientNetAntiSpoof, self).__init__()
        if arch == "b7":
            self.base = EfficientNet.from_name("efficientnet-b7")
        elif arch == "b5":
            self.base = EfficientNet.from_name("efficientnet-b5")
        elif arch == "b0":
            self.base = EfficientNet.from_name("efficientnet-b0")
        else:
            raise NotImplementedError

        if arch == "b7":
            self.base._fc = nn.Sequential(nn.Linear(in_features=2560, out_features=2, bias=True),
                                          nn.ReLU(inplace=True),
                                          )
        elif arch == "b5":
            self.base._fc = nn.Sequential(nn.Linear(in_features=2048, out_features=2, bias=True),
                                          nn.ReLU(inplace=True),
                                          )
        elif arch == "b0":
            self.base._fc = nn.Sequential(nn.Linear(in_features=1280, out_features=2, bias=True),
                                          nn.ReLU(inplace=True),
                                          )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.base(x)
        return x 
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        # for j in self.state_dict():
        #     print(j)
        for i in param_dict:
            # print(i)
            if 'module' in i:
                j = i.replace('module.','')
            else:
                j = i
            self.state_dict()[j].copy_(param_dict[i])
