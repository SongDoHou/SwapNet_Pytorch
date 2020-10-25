import torch
from torch import nn

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout=0.0):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # added by AJ
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class DualUNetUp(UNetUp):
    """
    My guess of how dual u-net works, according to the paper
    "Multi-View Image Generation from a Single-View"

    @author Andrew
    """

    def __init__(self, in_size, out_size, dropout=0.0):
        super(DualUNetUp, self).__init__(in_size, out_size, dropout)

    def forward(self, x, skip_input_1, skip_input_2):
        x = self.model(x)
        # print("DualUNetUp before cat:", x.shape)
        x = torch.cat((x, skip_input_1, skip_input_2), 1)

        return x