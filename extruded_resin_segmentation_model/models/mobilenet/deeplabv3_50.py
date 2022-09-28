from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import torch


class Deeplabv3(nn.Module):
    def __init__(self, n_class=21):
        super(Deeplabv3, self).__init__()
        self.n_class = n_class
        self.deeplabv3 = deeplabv3_mobilenet_v3_large(pretrained=True)
        self.deeplabv3.classifier = DeepLabHead(2048, self.n_class)
        

    def forward(self, x, debug=False):
        return self.deeplabv3(x)['out']

    def resume(self, file, test=False):
        if test and not file:
            self.deeplabv3 = deeplabv3_mobilenet_v3_large(pretrained=True, num_classes=self.n_class)
            return
        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file)
            print(f"best epoch : {checkpoint['epoch']}")
            checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)
