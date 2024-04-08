import torch
import torch.nn as nn
import math

class SELayer(nn.Moudle):
    """
    Squeeze and Exciation
    Only channel attention.
    """
    def __init__(self, channel, reduction=16, pool_mode='avg'):
        super().__init__()
        if pool_mode == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(1)

        self.full_connect = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.pooling(x).view(b,c)
        y = self.full_connect(y).view(b,c,1,1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.full_connect1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes//ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.full_connect2 = nn.Conv2d(in_channels=in_planes//ratio, out_channels=in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.full_connect1(self.avg_pooling(x))
        avg_out = self.relu(avg_out)
        avg_out = self.full_connect2(avg_out)

        max_out = self.full_connect1(self.max_pooling(x))
        max_out = self.relu(max_out)
        max_out = self.full_connect2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)

        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _  = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))

        return out

class CBAM(nn.Module):
    """
    CBAM: CAM and SAM
    contains both channel attention and spartial attention
    """
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes=in_channel, ratio=ratio)
        self.space_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.space_attention(x)

        return x


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = v.squeeze(-1).transpose(-1, -2)
        v = self.conv(v).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v
