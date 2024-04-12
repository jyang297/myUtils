import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # Backbone layers
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)

        # Lateral layers for feature pyramid
        self.lateral1 = nn.Conv2d(512, 256, 1)
        self.lateral2 = nn.Conv2d(256, 256, 1)
        self.lateral3 = nn.Conv2d(128, 256, 1)
        self.lateral4 = nn.Conv2d(64, 256, 1)

        # Upsampling and merging layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x1, x2):
        # Feature extraction from both images
        f1 = self.conv1(x1)
        f2 = self.conv1(x2)

        f1 = self.conv2(f1)
        f2 = self.conv2(f2)

        f1 = self.conv3(f1)
        f2 = self.conv3(f2)

        f1 = self.conv4(f1)
        f2 = self.conv4(f2)

        # Building the feature pyramid
        p1 = self.lateral1(f1)
        p2 = self.lateral2(f1) + self.upsample(p1)
        p3 = self.lateral3(f1) + self.upsample(p2)
        p4 = self.lateral4(f1) + self.upsample(p3)

        return p1, p2, p3, p4

# Example usage
fpn = FPN()
image1 = torch.randn(1, 3, 224, 224)
image2 = torch.randn(1, 3, 224, 224)
pyramid_features = fpn(image1, image2)

for feature in pyramid_features:
    print(feature.shape)
