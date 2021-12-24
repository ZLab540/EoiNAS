import torch
import torch.nn as nn
from operations import NaiveGN
import torch.nn.functional as F


class Original_ASPP(nn.Module):
    def __init__(self, C, depth, norm=NaiveGN, mult=1):
        super(Original_ASPP, self).__init__()
        self._C = C
        self._depth = depth

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.aspp1 = nn.Conv2d(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = nn.Conv2d(C, depth, kernel_size=3, stride=1, dilation=int(6 * mult), padding=int(6 * mult), bias=False)
        self.aspp3 = nn.Conv2d(C, depth, kernel_size=3, stride=1, dilation=int(12 * mult), padding=int(12 * mult), bias=False)
        self.aspp4 = nn.Conv2d(C, depth, kernel_size=3, stride=1, dilation=int(18 * mult), padding=int(18 * mult), bias=False)
        self.aspp5 = nn.Conv2d(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_gn = norm(8, depth)
        self.aspp2_gn = norm(8, depth)
        self.aspp3_gn = norm(8, depth)
        self.aspp4_gn = norm(8, depth)
        self.aspp5_gn = norm(8, depth)
        self.conv2 = nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, bias=False)
        self.gn2 = norm(8, depth)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_gn(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_gn(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_gn(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_gn(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_gn(x5)
        x5 = F.interpolate(x5, (x.shape[2], x.shape[3]), None, 'bilinear', True)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.gn2(x)

        return x

