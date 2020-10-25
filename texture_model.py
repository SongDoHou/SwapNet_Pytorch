import torch
import torch.nn as nn
from utils import *
from torchvision.ops import roi_align

class TextureModel(nn.Module):
    def __init__(self, texture_channels=3, cloth_channels=19, num_roi=12,
                 norm_type="batch", dropout=0.5, unet_type="pix2pix", img_size=128):
        super(TextureModel, self).__init__()
        # self.roi_align = ROIAlign(
        #     output_size=(128, 128), spatial_scale=1, sampling_ratio=1
        # )


        self.num_roi = num_roi
        channels = texture_channels * num_roi
        self.encode = UNetDown(channels, channels)
        self.unet = nn.Sequential(
            UNetDown(channels + cloth_channels, 64, normalize=False),
            UNetDown(64, 128),
            UNetDown(128, 256),
            UNetDown(256, 512, dropout=dropout),
            UNetDown(512, 1024, dropout=dropout),
            UNetDown(1024, 1024, normalize=False, dropout=dropout),
            UNetUp(1024, 1024, dropout=dropout),
            UNetUp(2 * 1024, 512, dropout=dropout),
            UNetUp(2 * 512, 256),
            UNetUp(2 * 256, 128),
            UNetUp(2 * 128, 64),
            # upsample and pad
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, texture_channels, 4, padding=1),
            nn.Tanh(),
        )
    @staticmethod
    def reshape_rois(rois):
        """
        Takes a (batch x num_rois x num_coordinates) and reshapes it into a 2D tensor.
        The 2D tensor has the first column as the batch index and the remaining columns
        as the coordinates.

        num_coordinates should be 4.

        :param rois: a (batch x num_rois x num_coordinates) tensor. coordinates is 4
        :return: a 2D tensor formatted for roi layers
        """
        # Append batch to rois
        # get the batch indices
        b_idx = torch.arange(rois.shape[0]).unsqueeze_(-1)
        # expand out and reshape to to batchx1 dimension
        b_idx = b_idx.expand(rois.shape[0], rois.shape[1]).reshape(-1).unsqueeze_(-1)
        b_idx = b_idx.to(rois.device).type(rois.dtype)
        reshaped = rois.view(-1, rois.shape[-1])
        reshaped = torch.cat((b_idx, reshaped), dim=1)
        return reshaped

    def forward(self, input_texture, rois, cloth):
        rois = TextureModel.reshape_rois(rois)
        pooled_rois = roi_align()#self.roi_align(input_texture, rois)
        # reshape the pooled rois such that pool output goes in the channels instead of
        # batch size
        batch_size = int(pooled_rois.shape[0] / self.num_roi)
        pooled_rois = pooled_rois.view(
            batch_size, -1, pooled_rois.shape[2], pooled_rois.shape[3]
        )

        encoded_tex = self.encode(pooled_rois)

        scale_factor = input_texture.shape[2] / encoded_tex.shape[2]
        upsampled_tex = nn.functional.interpolate(
            encoded_tex, scale_factor=scale_factor
        )
        tex_with_cloth = torch.cat((upsampled_tex, cloth), 1)

        return self.unet(tex_with_cloth)

if __name__ == "__main__":
    texture_model = TextureModel().cuda()
    # toy_texture =
    # toy_rois =
    # toy_cloth =