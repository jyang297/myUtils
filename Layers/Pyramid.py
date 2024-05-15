import torch.nn as nn


class FeaturePyramid(nn.Module):
    """Two-level feature pyramid
    1) remove high-level feature pyramid (compared to PWC-Net), and add more conv layers to stage 2;
    2) do not increase the output channel of stage 2, in order to keep the cost of corr volume under control.
    """
    def __init__(self):
        super().__init__()
        c = 24
        self.conv_stage1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
        self.conv_stage2 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, img):
        pyramid_layer_0 = self.conv_stage1(img)
        pyramid_layer_1 = self.conv_stage2(pyramid_layer_0)

        return [pyramid_layer_0, pyramid_layer_1]
