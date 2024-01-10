import torch.nn as nn
from ..layers import unetConv2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_channels = 3
        self.is_batchnorm=True

        filters = [64, 128, 256, 512, 1024]
        #filters = [128, 256, 512, 1024, 2048]

        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

    def forward(self, x):
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        low_level = conv2
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        return center, low_level