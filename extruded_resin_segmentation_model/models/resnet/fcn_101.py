from torchvision.models.segmentation import fcn_resnet101
from torchvision.models.segmentation.fcn import FCNHead
import torch.nn as nn
import torch

# Uses bilinear interpolation for upsampling
# https://github.com/pytorch/vision/blob/master/
# torchvision/models/segmentation/_utils.py

class FCN(nn.Module):
    def __init__(self, n_class=21):
        super(FCN, self).__init__()
        self.n_class = n_class
        self.fcn = fcn_resnet101(pretrained=True)
        self.fcn.classifier = FCNHead(2048, self.n_class)

    def forward(self, x, debug=False):
        return self.fcn(x)['out']

    def resume(self, file, test=False):
        if test and not file:
            self.fcn = fcn_resnet101(pretrained=True, num_classes=self.n_class)
            return
        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file)
            print(f"best epoch : {checkpoint['epoch']}")
            checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)
