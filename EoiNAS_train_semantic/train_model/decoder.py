import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import NaiveGN


class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes, GroupNorm=NaiveGN):
        super(Decoder, self).__init__()
        C_low = 48
        self.conv1 = nn.Conv2d(low_level_inplanes, C_low, 1, bias=False)
        self.gn1 = GroupNorm(8, 48)
        self.last_conv = nn.Sequential(nn.Conv2d(304,256, kernel_size=3, stride=1, padding=1, bias=False),
                                       GroupNorm(8, 256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       GroupNorm(8, 256),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.gn1(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x
