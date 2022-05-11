import torch.nn as nn
from backbone import resnet


class ResNet(nn.Module):
    def __init__(self, class_num):
        super(ResNet, self).__init__()
        self.class_num = class_num
        self.model = resnet.resnext50_32x4d(pretrained=True)
        self.fc = nn.Linear(1000, 3)

    def forward(self, x):

        # feature shape: 512*4, classnum
        feature = self.model(x)
        output = self.fc(feature)

        return output