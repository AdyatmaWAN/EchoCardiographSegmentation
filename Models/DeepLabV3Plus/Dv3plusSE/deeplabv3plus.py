import torch.nn as nn
import torch.nn.functional as F
from .aspp import ASPP
from .Decoder.decoder import Decoder
from .Encoder.encoder import Encoder

class DeepLab(nn.Module):
    def __init__(self, output_stride=16, num_classes=1):
        super(DeepLab, self).__init__()       
        BatchNorm = nn.BatchNorm2d
        self.backbone = Encoder()
        self.aspp = ASPP(output_stride, BatchNorm)
        self.decoder = Decoder(num_classes, BatchNorm)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x