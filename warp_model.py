import torch
import torch.nn as nn
from utils import *

class WarpModel(nn.Module):
    """
    The warping module takes a body segmentation to represent the "pose",
    and an input clothing segmentation to transform to match the pose.
    """

    def __init__(self, body_channels=3, cloth_channels=19, dropout=0.5):
        super(WarpModel, self).__init__()
        self.body_down1 = UNetDown(body_channels, 64, normalize=False)
        self.body_down2 = UNetDown(64, 128)
        self.body_down3 = UNetDown(128, 256)
        self.body_down4 = UNetDown(256, 512, dropout=dropout)


        self.cloth_down1 = UNetDown(cloth_channels, 64, normalize=False)
        self.cloth_down2 = UNetDown(64, 128)
        self.cloth_down3 = UNetDown(128, 256)
        self.cloth_down4 = UNetDown(256, 512)
        self.cloth_down5 = UNetDown(512, 1024, dropout=dropout)
        self.cloth_down6 = UNetDown(1024, 1024, normalize=False, dropout=dropout)
        # the two UNetUp's below will be used WITHOUT concatenation.
        # hence the input size will not double
        self.cloth_up1 = UNetUp(1024, 1024)
        self.cloth_up2 = UNetUp(1024, 512)

        self.resblocks = nn.Sequential(
            # I don't really know if dropout should go here. I'm just guessing
            ResidualBlock(1024, dropout=dropout),
            ResidualBlock(1024, dropout=dropout),
            ResidualBlock(1024, dropout=dropout),
            ResidualBlock(1024, dropout=dropout),
        )

        # input encoded (512) & cat body_d4 (512) cloth_d4 (512)
        self.dual_up1 = DualUNetUp(1024, 256)
        # input dual_up1 (256) & cat body_d3 (256) cloth_d3 (256)
        self.dual_up2 = DualUNetUp(3 * 256, 128)
        # input dual_up2 (128) & cat body_d2 (128) cloth_d2 (128)
        self.dual_up3 = DualUNetUp(3 * 128, 64)

        self.upsample_and_pad = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(3 * 64, cloth_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, body, cloth):
        body_d1 = self.body_down1(body)
        body_d2 = self.body_down2(body_d1)
        body_d3 = self.body_down3(body_d2)
        body_d4 = self.body_down4(body_d3)

        cloth_d1 = self.cloth_down1(cloth)
        cloth_d2 = self.cloth_down2(cloth_d1)
        cloth_d3 = self.cloth_down3(cloth_d2)
        cloth_d4 = self.cloth_down4(cloth_d3)
        cloth_d5 = self.cloth_down5(cloth_d4)
        cloth_d6 = self.cloth_down6(cloth_d5)
        cloth_u1 = self.cloth_up1(cloth_d6, None)
        cloth_u2 = self.cloth_up2(cloth_u1, None)

        body_and_cloth = torch.cat((body_d4, cloth_u2), dim=1)
        encoded = self.resblocks(body_and_cloth)

        dual_u1 = self.dual_up1(encoded, body_d3, cloth_d3)
        dual_u2 = self.dual_up2(dual_u1, body_d2, cloth_d2)
        dual_u3 = self.dual_up3(dual_u2, body_d1, cloth_d1)

        upsampled = self.upsample_and_pad(dual_u3)
        return upsampled

if __name__ == "__main__":
    warp_model = WarpModel().cuda()
    toy_body = torch.randn((1, 3, 128, 128)).cuda()
    toy_cloth = torch.randn((1, 19, 128, 128)).cuda()

    result = warp_model(toy_body, toy_cloth)
    print(result.size()) # torch.Size([1, 19, 128, 128])