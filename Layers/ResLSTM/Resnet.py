import torch
import torch.nn as nn
import torchvision.models as models


class BasicResNetUnit(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x) :
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return  out

class NoReluBasicResNetUnit(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x) :
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        
        return  out


class ResNetLayer(nn.Module):
    '''
    The ResNetLayer is and only consisted of 4 residual layers. 
    Does not care about something like image extractor or one hot coding
    
    The input:
    block: The conv unit used to implement one of the two CNN blocks inside ResBlock
    layers: list consit of 4 values. It controls the layer numbers for each layer. Default is [2,2,2,2] for ResNet18(8*2+2)
    But this time 
    '''
    def __init__(self, block, layers:list):
        super().__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != block.expansion * out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, block.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = block.expansion * out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        '''
        The input x is feature rather than image. The in_planes need to be set when instancing
        '''        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

class ScaledResNet(nn.Module):
    from typing import Type, Callable
    """
    A Residual Block class for neural networks that includes optional batch normalization and
    activation function, and uses a residual scaling factor. This block is typically used
    in architectures like ResNets.

    Parameters:
        conv_layer (Type[nn.Module]): The convolutional layer class to use, e.g., nn.Conv2d.
        num_features (int): Number of input and output feature maps.
        kernel_size (int): Size of the convolution kernel.
        bias (bool, optional): Whether to include bias in convolution layers. Defaults to True.
        batch_norm (bool, optional): Whether to include batch normalization. Defaults to False.
        activation (Callable[[bool], nn.Module], optional): Activation function to use. Defaults to ReLU.
        res_scale (float, optional): Scaling factor for the residual connection. Defaults to 1.
    """

    def __init__(
            self,
            conv_layer: Type[nn.Module] = nn.Conv2d,
            num_features: int = 64,
            kernel_size: int = 3,
            bias: bool = True,
            batch_norm: bool = False,
            activation: Callable = lambda: nn.ReLU(True),
            res_scale: float = 1
    ):
        super().__init__()
        layers = []
        for i in range(2):
            layers.append(conv_layer(num_features, num_features, kernel_size, padding=kernel_size // 2, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features))
            if i == 0:  # Add activation only after the first convolution layer
                layers.append(activation())

        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res
